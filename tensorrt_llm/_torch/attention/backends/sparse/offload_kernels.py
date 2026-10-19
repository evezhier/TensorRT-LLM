# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kernels shared by sparse attention models that offload KV history.

A decode step's selection names units of a request's history; the KV cache
manager fetches them into per-request arena rows. These kernels mask a
selection to the history the Batch published, and assemble the arena rows the
attention reads from the fetch output and the in-flight unit's write row.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def check_sparse_offload_tensor(
    tensor: torch.Tensor, name: str, ndim: int, device: torch.device
) -> None:
    """Require a CUDA int32 tensor of ``ndim`` dimensions on ``device``."""
    if tensor.dtype != torch.int32 or tensor.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D int32 tensor")
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")


@triton.jit(do_not_specialize=["num_lengths"])
def _select_sparse_history_units_kernel(
    topk_ptr,
    request_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    out_stride0,
    out_stride1,
    NUM_REQUESTS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_rows_ptr + query * request_rows_stride)
    active = (query < active_count) & (request >= 0) & (request < active_count)
    active = active & (request < NUM_REQUESTS) & (request < num_lengths)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, active, other=0)
    history = tl.load(history_lengths_ptr + request * history_lengths_stride, active, other=0)
    k = tl.arange(0, BLOCK_K)
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, k < TOPK, other=-1)
    valid = (unit >= 0) & (unit < length) & (unit < history // UNIT_TOKENS)
    # Each live query owns a distinct Batch row. Keep top-k positions, including
    # holes left by in-flight units, so fetch output has the same slot meaning.
    tl.store(
        out_ptr + request * out_stride0 + k * out_stride1,
        tl.where(valid, unit, -1),
        mask=active & (k < TOPK),
    )


def select_sparse_history_units(
    topk_indices: torch.Tensor,
    request_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
) -> None:
    """Mask non-history selections and put query rows into native Batch order.

    All tensors are CUDA int32 on one device. ``topk_indices`` is [Q, K] in
    selection units, with distinct nonnegative selections per live row.
    ``request_rows`` [Q] maps live queries to a permutation of the active Batch
    prefix, one query per request. ``active_request_count`` [1] excludes graph
    padding. ``unit_lengths`` [R] counts each request's selectable units, in
    request order, and may cover fewer rows than the native Batch capacity B.
    ``history_lengths`` [B] is the Batch-owned, published history watermark in
    original tokens.

    ``out`` is caller-owned [B, K], disjoint from the inputs. Only ordinals below
    both the unit length and history_lengths // unit_tokens are fetched.
    Top-k order and padding positions are preserved; in-flight units and invalid
    selections become -1. Unused Batch rows are overwritten with -1 on every
    call, including when a graph's Q is smaller than B. No tensor allocation,
    host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_rows, "request_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_requests, width = out.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if topk != width:
        raise ValueError("top-k width must match the configured sparse selection width")
    if request_rows.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_rows must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_requests,):
        raise ValueError("history_lengths must cover the native Batch capacity")
    if not num_queries <= unit_lengths.numel() <= num_requests:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")

    with torch.cuda.device(device):
        # Clear separately: padded queries can carry stale or invalid request IDs,
        # and a smaller graph may leave native Batch rows without any query.
        out.fill_(-1)
        if num_queries == 0 or topk == 0:
            return
        _select_sparse_history_units_kernel[(num_queries,)](
            topk_indices,
            request_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            out,
            *topk_indices.stride(),
            request_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *out.stride(),
            NUM_REQUESTS=num_requests,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )


@triton.jit(do_not_specialize=["num_lengths"])
def _merge_sparse_arena_indices_kernel(
    topk_ptr,
    request_rows_ptr,
    active_request_count_ptr,
    unit_lengths_ptr,
    history_lengths_ptr,
    fetched_ptr,
    write_ptr,
    out_ptr,
    topk_stride0,
    topk_stride1,
    request_rows_stride,
    unit_lengths_stride,
    history_lengths_stride,
    fetched_stride0,
    fetched_stride1,
    write_stride,
    out_stride0,
    out_stride1,
    NUM_REQUESTS: tl.constexpr,
    num_lengths,
    TOPK: tl.constexpr,
    UNIT_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query = tl.program_id(0)
    active_count = tl.load(active_request_count_ptr)
    request = tl.load(request_rows_ptr + query * request_rows_stride)
    live = (query < active_count) & (request >= 0) & (request < active_count)
    live = live & (request < NUM_REQUESTS) & (request < num_lengths)
    length = tl.load(unit_lengths_ptr + request * unit_lengths_stride, live, other=0)
    history_units = (
        tl.load(history_lengths_ptr + request * history_lengths_stride, live, other=0)
        // UNIT_TOKENS
    )
    k = tl.arange(0, BLOCK_K)
    in_width = k < TOPK
    unit = tl.load(topk_ptr + query * topk_stride0 + k * topk_stride1, in_width, other=-1)
    valid = live & (unit >= 0) & (unit < length)
    fetched = tl.load(
        fetched_ptr + request * fetched_stride0 + k * fetched_stride1,
        valid & in_width & (unit < history_units),
        other=-1,
    )
    write = tl.load(write_ptr + request * write_stride, live, other=-1)
    tl.store(
        out_ptr + query * out_stride0 + k * out_stride1,
        tl.where(valid & (unit == history_units), write, fetched),
        mask=in_width,
    )


def merge_sparse_arena_indices(
    topk_indices: torch.Tensor,
    request_rows: torch.Tensor,
    active_request_count: torch.Tensor,
    unit_lengths: torch.Tensor,
    history_lengths: torch.Tensor,
    fetched_indices: torch.Tensor,
    write_indices: torch.Tensor,
    out: torch.Tensor,
    *,
    unit_tokens: int,
) -> torch.Tensor:
    """Return each query's selection as rows of one sparse buffer's arena.

    Inputs follow :func:`select_sparse_history_units`. ``fetched_indices`` is
    the fetch output of one buffer, [B, K] in Batch row order and top-k column
    order; ``write_indices`` [B] is that buffer's in-flight row, written this
    step. Both may be strided views of the per-buffer outputs.

    ``out`` is [Q, K] in query order. A selected history unit takes its fetched
    row, the in-flight unit takes the write row, and padding queries, invalid
    selections and units past the unit length take -1. No tensor allocation,
    host read, or synchronization is performed.
    """
    device = out.device
    for tensor, name, ndim in (
        (topk_indices, "topk_indices", 2),
        (request_rows, "request_rows", 1),
        (active_request_count, "active_request_count", 1),
        (unit_lengths, "unit_lengths", 1),
        (history_lengths, "history_lengths", 1),
        (fetched_indices, "fetched_indices", 2),
        (write_indices, "write_indices", 1),
        (out, "out", 2),
    ):
        check_sparse_offload_tensor(tensor, name, ndim, device)
    num_requests, width = fetched_indices.shape
    num_queries, topk = topk_indices.shape
    if unit_tokens <= 0:
        raise ValueError("unit_tokens must be positive")
    if topk != width or out.shape != (num_queries, topk):
        raise ValueError("Selections, fetch output and out must share the top-k width")
    if request_rows.shape != (num_queries,) or active_request_count.shape != (1,):
        raise ValueError("request_rows must be [Q] and active_request_count must be [1]")
    if history_lengths.shape != (num_requests,) or write_indices.shape != (num_requests,):
        raise ValueError("history_lengths and write_indices must cover the native Batch")
    if not num_queries <= unit_lengths.numel() <= num_requests:
        raise ValueError("Query and unit-length rows must fit the native Batch capacity")
    if num_queries == 0 or topk == 0:
        return out
    with torch.cuda.device(device):
        _merge_sparse_arena_indices_kernel[(num_queries,)](
            topk_indices,
            request_rows,
            active_request_count,
            unit_lengths,
            history_lengths,
            fetched_indices,
            write_indices,
            out,
            *topk_indices.stride(),
            request_rows.stride(0),
            unit_lengths.stride(0),
            history_lengths.stride(0),
            *fetched_indices.stride(),
            write_indices.stride(0),
            *out.stride(),
            NUM_REQUESTS=num_requests,
            num_lengths=unit_lengths.numel(),
            TOPK=topk,
            UNIT_TOKENS=unit_tokens,
            BLOCK_K=triton.next_power_of_2(topk),
        )
    return out
