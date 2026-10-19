# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the kernels that assemble sparse offload arena rows."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import (
    merge_sparse_arena_indices,
    select_sparse_history_units,
)

_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _cuda_tensor(value, strided: bool = False) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.int32, device="cuda")
    if not strided:
        return tensor
    storage = torch.full(
        tuple(2 * n for n in tensor.shape), 12345, dtype=torch.int32, device="cuda"
    )
    view = storage[tuple(slice(None, None, 2) for _ in tensor.shape)]
    view.copy_(tensor)
    return view


def _merge_reference(topk, rows, active, lengths, history, fetched, writes, unit_tokens):
    out = torch.full(topk.shape, -1, dtype=torch.int32)
    for query, row in enumerate(rows.tolist()):
        if query >= active or not 0 <= row < min(active, history.numel(), lengths.numel()):
            continue
        history_units = history[row].item() // unit_tokens
        for k, unit in enumerate(topk[query].tolist()):
            if not 0 <= unit < lengths[row].item():
                continue
            if unit < history_units:
                out[query, k] = fetched[row, k]
            elif unit == history_units:
                out[query, k] = writes[row]
    return out


@_requires_cuda
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("active", [0, 2, 3])
def test_merge_takes_fetched_history_and_the_write_row(strided, active):
    # One-token units: a decode step's kv length is its history plus the token
    # being written, and that token's ordinal equals the history.
    history = torch.tensor([7, 0, 12, 99], dtype=torch.int32)
    lengths = history[:3] + 1
    rows = torch.tensor([2, 0, 1], dtype=torch.int32)
    topk = torch.tensor(
        [[12, 0, -1, 11, 13], [7, 6, 0, 8, -1], [0, 1, -1, -1, -1]], dtype=torch.int32
    )
    fetched = torch.arange(4 * 5, dtype=torch.int32).reshape(4, 5) + 100
    writes = torch.tensor([40, 41, 42, 43], dtype=torch.int32)
    out = torch.full((3, 5), 999, dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        _cuda_tensor(topk, strided),
        _cuda_tensor(rows, strided),
        _cuda_tensor([active]),
        _cuda_tensor(lengths, strided),
        _cuda_tensor(history, strided),
        _cuda_tensor(fetched, strided),
        _cuda_tensor(writes, strided),
        out,
        unit_tokens=1,
    )
    expected = _merge_reference(topk, rows, active, lengths, history, fetched, writes, 1)
    torch.testing.assert_close(out.cpu(), expected)
    if active == 3:
        # Request 2 has history 12, so ordinal 12 is the token being written.
        assert out[0, 0].item() == 42
        # Ordinal 13 is past the kv length.
        assert out[0, 4].item() == -1


@_requires_cuda
@pytest.mark.parametrize("topk_size", [1, 7, 2048])
def test_merge_matches_selection_and_fetch_order(topk_size):
    generator = torch.Generator().manual_seed(7)
    history = torch.tensor([3000, 41, 1, 0], dtype=torch.int32)
    lengths = history[:3] + 1
    rows = torch.tensor([1, 2, 0], dtype=torch.int32)
    topk = torch.stack(
        [torch.randperm(3001, generator=generator)[:topk_size] for _ in range(3)]
    ).to(torch.int32)
    topk[:, 2::7] = -1
    selected = torch.empty((4, topk_size), dtype=torch.int32, device="cuda")
    select_sparse_history_units(
        _cuda_tensor(topk),
        _cuda_tensor(rows),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(history),
        selected,
        unit_tokens=1,
    )
    # A fetch returns an arena row for every selected history unit, in slot order.
    fetched = torch.where(selected >= 0, selected * 3 + 1, -1)
    writes = _cuda_tensor([5, 6, 7, 8])
    out = torch.empty((3, topk_size), dtype=torch.int32, device="cuda")
    merge_sparse_arena_indices(
        _cuda_tensor(topk),
        _cuda_tensor(rows),
        _cuda_tensor([3]),
        _cuda_tensor(lengths),
        _cuda_tensor(history),
        fetched,
        writes,
        out,
        unit_tokens=1,
    )
    expected = _merge_reference(
        topk, rows, 3, lengths, history, fetched.cpu(), writes.cpu(), unit_tokens=1
    )
    torch.testing.assert_close(out.cpu(), expected)


@_requires_cuda
def test_merge_replays_in_a_cuda_graph_with_new_watermarks():
    topk = _cuda_tensor([[0, 1, 2, 3], [0, 1, -1, -1]])
    rows = _cuda_tensor([0, 1])
    active = _cuda_tensor([2])
    history = _cuda_tensor([2, 1, 0])
    lengths = _cuda_tensor([3, 2])
    fetched = _cuda_tensor([[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]])
    writes = _cuda_tensor([7, 8, 9])
    out = torch.empty((2, 4), dtype=torch.int32, device="cuda")
    args = (topk, rows, active, lengths, history, fetched, writes, out)
    merge_sparse_arena_indices(*args, unit_tokens=1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        merge_sparse_arena_indices(*args, unit_tokens=1)
    for hist0, length0, active_count, expected0 in (
        (2, 3, 2, [10, 11, 7, -1]),
        (3, 4, 2, [10, 11, 12, 7]),
        (0, 1, 2, [7, -1, -1, -1]),
        (3, 4, 0, [-1, -1, -1, -1]),
    ):
        history[0] = hist0
        lengths[0] = length0
        active[0] = active_count
        graph.replay()
        torch.testing.assert_close(out[0].cpu(), torch.tensor(expected0, dtype=torch.int32))


@_requires_cuda
@pytest.mark.parametrize("invalid", ["dtype", "width", "rows", "writes", "lengths", "unit"])
def test_merge_rejects_invalid_layout(invalid):
    topk = _cuda_tensor([[0, 1], [1, 0]])
    rows, active, lengths, history = (
        _cuda_tensor([0, 1]),
        _cuda_tensor([2]),
        _cuda_tensor([2, 2]),
        _cuda_tensor([1] * 4),
    )
    fetched = _cuda_tensor([[0, 0]] * 4)
    writes = _cuda_tensor([0] * 4)
    unit = 1
    if invalid == "dtype":
        topk = topk.to(torch.int64)
    elif invalid == "width":
        fetched = _cuda_tensor([[0]] * 4)
    elif invalid == "rows":
        rows = rows[:1]
    elif invalid == "writes":
        writes = writes[:2]
    elif invalid == "lengths":
        lengths = lengths[:1]
    else:
        unit = 0
    with pytest.raises(ValueError):
        merge_sparse_arena_indices(
            topk,
            rows,
            active,
            lengths,
            history,
            fetched,
            writes,
            torch.empty((2, 2), dtype=torch.int32, device="cuda"),
            unit_tokens=unit,
        )
