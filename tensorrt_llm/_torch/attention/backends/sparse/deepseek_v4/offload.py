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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import Batch, BufferId, LayerGroupId


@dataclass(frozen=True, slots=True)
class SparseOffloadLayerDescriptor:
    """All sparse buffers of one model layer, in KVCM's required fetch order.

    ``unit_tokens`` original tokens produce one compressed row. Fetch returns
    a separate physical index for each buffer, relative to that buffer's
    ``get_sparse_cache_base_address()``. These indices already include the
    request and buffer placement; no page scaling or layer offset is needed.
    """

    buffer_ids: tuple[BufferId, ...]
    group_id: LayerGroupId
    unit_tokens: int


@dataclass(kw_only=True)
class SparseOffloadState:
    """Fixed-capacity storage owned by one serial attention metadata instance.

    All device tensors are int32. Selections are [B, K], fetch outputs are
    [B, F, K], and the active count is [1]. B is the native Batch capacity,
    including when an attention graph uses fewer query rows. Output storage
    is shared by layers with the same buffer count F. Reuse must follow the
    previous layer's attention on the consuming stream.

    ``history_lengths`` is a zero-copy view of Batch's published token counts,
    bound outside capture. Batch owns the page tables and watermarks; metadata
    does not maintain another copy. Query-to-Batch row mappings are supplied
    to selection on the GPU. Fetch output remains in Batch row order.

    ``prepared`` means preparation was enqueued successfully; stream ordering
    establishes device readiness. ``is_prefill`` is set by host batch validation
    before execution, never inferred from a device count inside a layer.

    An in-flight selection is masked out of the history fetch. Its physical
    index must come from the separately prepared write mapping when assembling
    attention inputs.
    """

    layers: dict[int, SparseOffloadLayerDescriptor]
    active_request_count: torch.Tensor
    selected_history_units: torch.Tensor
    fetched_indices: dict[int, torch.Tensor]
    batch: Batch | None = None
    history_lengths: torch.Tensor | None = None
    prepared: bool = False
    is_prefill: bool = False
