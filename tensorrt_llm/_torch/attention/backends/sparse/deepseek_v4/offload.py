# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention-owned descriptors and persistent sparse offload metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Collection

import torch

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import Batch, BufferId, LayerGroupId


@dataclass(frozen=True, slots=True)
class SparseOffloadPaddingRegion:
    """Unused buffer bytes across arena leases, initialized during manager construction."""

    base_address: int
    bytes_per_lease: int
    lease_stride: int


@dataclass(frozen=True, slots=True)
class SparseOffloadLayerDescriptor:
    """All sparse buffers of one model layer, in KVCM's required fetch order.

    ``unit_tokens`` original tokens produce one compressed row. Fetch returns
    a separate physical index for each buffer, relative to that buffer's
    ``get_sparse_cache_base_address()``. These indices already include the
    request and buffer placement; no page scaling or layer offset is needed.
    Descriptor construction validates matching NVFP4 data/scale pool counts
    and buffer positions, including padding. Their index columns are equal,
    so compressor scatter and gather consume the data column for both pools
    while retaining their respective base addresses and byte strides.
    ``arena_row_counts`` covers each complete pool region across all leases
    and buffers, including padding; it is independent of host-page capacity.
    """

    buffer_ids: tuple[BufferId, ...]
    arena_base_addresses: tuple[int, ...]
    arena_row_counts: tuple[int, ...]
    data_buffer_index: int
    scale_buffer_index: int | None
    group_id: LayerGroupId
    unit_tokens: int
    padding_regions: tuple[SparseOffloadPaddingRegion, ...] = ()


@dataclass(kw_only=True)
class SparseOffloadState:
    """Fixed-capacity storage owned by one serial attention metadata instance.

    Index and count tensors are int32. Selections are [B, K], fetch outputs are
    [B, F, K], write destinations are [B, F, 1], and the active count is [1].
    B is the native Batch capacity, including when an attention graph uses
    fewer query rows. The existing attention-index preparation kernel reads
    fetch/write mappings directly to assemble the query-ordered index list. Storage
    is shared by layers with the same buffer count F. Reuse must follow the
    previous layer's attention on the consuming stream.

    NVFP4 additionally uses persistent FP8 scratch [B, K, head_dim] for the
    dequantized rows consumed by attention. Arena allocations remain owned by
    KVCM; the cache manager retains non-owning tensor views sized by arena
    geometry. ``scatter_start_positions`` is initialized to zero once and
    supplies the one-row-page view used by the existing compressor scatter.

    ``history_lengths`` is a zero-copy view of Batch's published token counts,
    bound outside capture. Batch owns the page tables and watermarks; metadata
    does not maintain another copy. Query-to-Batch row mappings are supplied
    to selection on the GPU. Fetch output remains in Batch row order.

    ``prepared`` means preparation was enqueued successfully; stream ordering
    establishes device readiness. ``is_prefill`` is set by host batch validation
    before execution, never inferred from a device count inside a layer.
    ``write_layer`` tracks which layer enqueued the shared write mapping during
    eager execution or capture. Graph replay uses the captured GPU operations;
    this host marker does not determine device readiness.

    An in-flight selection is masked out of the history fetch. Its physical
    index must come from the separately prepared write mapping when assembling
    attention inputs.
    """

    layers: dict[int, SparseOffloadLayerDescriptor]
    active_request_count: torch.Tensor
    selected_history_units: torch.Tensor
    fetched_indices: dict[int, torch.Tensor]
    write_indices: dict[int, torch.Tensor]
    scatter_start_positions: torch.Tensor
    nvfp4_scratch: torch.Tensor | None = None
    batch: Batch | None = None
    history_lengths: torch.Tensor | None = None
    prepared: bool = False
    is_prefill: bool = False
    write_layer: int | None = None

    @staticmethod
    def device_bytes(
        *,
        max_batch_size: int,
        topk: int,
        buffer_counts: Collection[int],
        nvfp4_head_dim: int | None,
    ) -> int:
        """Return the device bytes one instance allocates.

        Mirrors the allocations of the attention metadata's
        ``_init_sparse_offload_state``.

        Args:
            max_batch_size: Rows B, the native Batch capacity.
            topk: Units one row selects, K.
            buffer_counts: The distinct sparse buffer counts F of the layers.
            nvfp4_head_dim: Head dim of the dequantized rows with NVFP4, else None.
        """
        # The active count, the selections and the scatter start positions, then
        # the fetch outputs and write destinations of each buffer count.
        index_entries = 1 + max_batch_size * topk + max_batch_size
        index_entries += sum(max_batch_size * count * (topk + 1) for count in buffer_counts)
        nvfp4_bytes = 0
        if nvfp4_head_dim is not None:
            nvfp4_bytes = max_batch_size * topk * nvfp4_head_dim * torch.float8_e4m3fn.itemsize
        return index_entries * torch.int32.itemsize + nvfp4_bytes
