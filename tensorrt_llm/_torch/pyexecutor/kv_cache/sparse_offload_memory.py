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

"""Memory accounting for sparse attention KV cache offload.

With offload, the KV history a sparse attention layer selects from lives in
host-backed pages, and each request leases a GPU cache of selected units from a
sparse cell arena.

The native manager puts a sparse buffer's pages in a host-backed pool of the
same pool group as its life cycle's dense buffers. A slot spans every pool of
its group, so the host-backed pools get the slot count the device quota buys
for the dense ones, and nothing else bounds them. The host memory offload takes
is therefore set by the device quota and by how that quota is split across life
cycles. The arena is a separate device allocation outside every tier's quota.

Everything here is derived from the layer configs a cache manager declares, so
it is the same for every model. A model marks each offloaded buffer
``is_sparse``, gives its layer a fetch path from :func:`sparse_cache_config`,
and reports the device bytes of its other buffers through
``KVCacheManagerV2.get_request_device_bytes``.

The module has three parts: mirrors of the native layout, which must change
when ``cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2`` does; the sizing
policy; and the capacity report and host admission.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.runtime.kv_cache_manager_v2 import AttentionLayerConfig, SparseCacheConfig

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BufferConfig,
        KVCacheManagerConfig,
        LayerConfig,
    )

_GIB = float(1 << 30)
_MIB = float(1 << 20)

# ---------------------------------------------------------------------------
# Native layout: SparseCellArena and leaseLayout in sparseCache.cpp, the Batch in
# batch.cpp, pool and life-cycle formation in storage/config.cpp and
# lifeCycleRegistry.cpp, cacheTierGranularity in storageManager.cpp, and HostMem
# in utils/hostMem.cpp.
# ---------------------------------------------------------------------------

# SparseCellArena rounds every region to this many bytes.
SPARSE_ARENA_ALIGNMENT = 256
# Every arena and Batch index array holds 32-bit entries (int32_t or uint32_t).
_INDEX_BYTES = 4
# A native SparseCacheView takes at most this many bytes; a lease aligns it to this size.
_SPARSE_CACHE_VIEW_BYTES = 256
# A Batch row holds two watermarks, history length and capacity.
_BATCH_WATERMARKS_PER_ROW = 2
# A Batch starts its cache views at a pointer-aligned offset.
_BATCH_VIEW_ALIGNMENT = 8
# Pinned host memory is mapped in whole pages of this many bytes.
_HOST_PAGE_BYTES = 4096
# A decode step writes one query token per request, which completes at most one
# unit, so one unit per layer is in flight (sparseCacheMaxInFlightUnits).
_SPARSE_MAX_FLUSH_UNITS = 1
# The GPU tier allocates pools in granules of 2 MiB, doubled for every doubling
# of the quota above 1 GiB, at most four times.
_GPU_GRANULE_BASE_BYTES = 2 << 20
_GPU_GRANULE_MAX_DOUBLINGS = 4


def _index_array_bytes(num_entries: int) -> int:
    """Return the aligned bytes of an arena array of ``num_entries`` 32-bit entries."""
    return pad_up(_INDEX_BYTES * num_entries, SPARSE_ARENA_ALIGNMENT)


def gpu_pool_granule_bytes(quota: int) -> int:
    """Return the granule the native GPU tier allocates the pools of a ``quota``-byte tier in."""
    quota_gib = quota // (1 << 30)
    doublings = 0 if quota_gib == 0 else min(_GPU_GRANULE_MAX_DOUBLINGS, quota_gib.bit_length() - 1)
    return _GPU_GRANULE_BASE_BYTES << doublings


def sparse_units_per_page(unit_tokens: int | None, tokens_per_block: int) -> int:
    """Return the selection units one page holds.

    Args:
        unit_tokens: Tokens one unit covers, or None when a unit is a whole page.
        tokens_per_block: Tokens per page.
    """
    return tokens_per_block // (unit_tokens or tokens_per_block)


@dataclass(frozen=True, slots=True)
class SparseArenaShape:
    """GPU cell arena of one sparse life cycle.

    Attributes:
        num_layers: Layers with a sparse fetch path; each has one metadata row.
        cells_per_buffer: Cells each fetched sparse buffer gets per lease.
        units_per_cell: Selection units in one cell.
        cell_bytes: Bytes of one cell of each fetched sparse buffer, in layer order.
    """

    num_layers: int
    cells_per_buffer: int
    units_per_cell: int
    cell_bytes: tuple[int, ...]

    @property
    def entries(self) -> int:
        """Cache entries of one lease: one per cached unit of each layer."""
        return self.num_layers * self.cells_per_buffer * self.units_per_cell

    @property
    def lease_metadata_bytes(self) -> int:
        """Bytes of one lease's recency, residency and write-back state."""
        flush_entries = self.num_layers * _SPARSE_MAX_FLUSH_UNITS
        max_flush_blocks = ceil_div(_SPARSE_MAX_FLUSH_UNITS, self.units_per_cell) + 1
        arrays = (
            self.entries,  # residentOrdinal
            self.entries,  # lruOrder
            flush_entries,  # flushOrdinal
            flush_entries,  # flushSlot
            self.num_layers,  # flushCount
            2,  # flushRange
            max_flush_blocks,  # flushPageSlice
        )
        return sum(_index_array_bytes(n) for n in arrays) + _SPARSE_CACHE_VIEW_BYTES

    @property
    def lease_bytes(self) -> int:
        """GPU bytes one lease adds to the arena."""
        return self.cells_per_buffer * sum(self.cell_bytes) + self.lease_metadata_bytes

    def arena_bytes(self, max_leases: int) -> int:
        """Return the GPU bytes of an arena sized for ``max_leases`` leases."""
        # One region per pool. Buffers of one cell size share a pool, so a region
        # holds every lease's cells of every buffer of that size.
        buffers_by_cell_bytes: dict[int, int] = defaultdict(int)
        for cell_bytes in self.cell_bytes:
            buffers_by_cell_bytes[cell_bytes] += 1
        cell_regions = sum(
            pad_up(
                max_leases * num_buffers * self.cells_per_buffer * cell_bytes,
                SPARSE_ARENA_ALIGNMENT,
            )
            for cell_bytes, num_buffers in buffers_by_cell_bytes.items()
        )
        num_buffers = len(self.cell_bytes)
        shared_arrays = (
            num_buffers,  # page offset of each buffer
            num_buffers,  # metadata row of each buffer
            num_buffers,  # slot of each buffer in its pool
            self.entries,  # recency order a lease starts from
        )
        shared = sum(_index_array_bytes(n) for n in shared_arrays)
        return cell_regions + shared + max_leases * self.lease_metadata_bytes


def _fetched_sparse_buffers(layer: LayerConfig) -> list[BufferConfig]:
    """Return the sparse buffers of ``layer`` that are read through a sparse cache."""
    if getattr(layer, "sparse_cache", None) is None:
        return []
    return [buffer for buffer in layer.buffers if buffer.is_sparse]


def _fetch_path_life_cycles(
    layers: Sequence[LayerConfig], tokens_per_block: int
) -> list[list[AttentionLayerConfig]]:
    """Return the layers with a sparse fetch path, grouped by life cycle."""
    # Layers share a life cycle when their window and sink pages match.
    groups: dict[tuple[int | None, int], list[AttentionLayerConfig]] = defaultdict(list)
    for layer in layers:
        if _fetched_sparse_buffers(layer):
            sink_blocks = ceil_div(layer.num_sink_tokens or 0, tokens_per_block)
            groups[layer.sliding_window_size, sink_blocks].append(layer)
    return list(groups.values())


def sparse_arena_shapes(
    layers: Sequence[LayerConfig], *, tokens_per_block: int
) -> list[SparseArenaShape]:
    """Return the arena of every life cycle whose layers have a sparse fetch path.

    Args:
        layers: Every layer config the manager allocates.
        tokens_per_block: Tokens per page.

    Raises:
        ValueError: Layers sharing a life cycle declare different
            ``sparse_cache`` values. They share one arena, so the native manager
            rejects them too.
    """
    shapes = []
    for grouped_layers in _fetch_path_life_cycles(layers, tokens_per_block):
        geometries = {
            (
                layer.sparse_cache.num_selected_blocks,
                layer.sparse_cache.max_cached_units,
                sparse_units_per_page(layer.sparse_cache.unit_tokens, tokens_per_block),
            )
            for layer in grouped_layers
        }
        if len(geometries) > 1:
            raise ValueError(
                "Layers sharing a life cycle must declare the same sparse_cache; they share "
                f"one arena. Got (selected, cached, units per page) {sorted(geometries)}."
            )
        ((_, max_cached_units, units_per_page),) = geometries
        shapes.append(
            SparseArenaShape(
                num_layers=len(grouped_layers),
                cells_per_buffer=ceil_div(max_cached_units, units_per_page),
                units_per_cell=units_per_page,
                # A cell holds one page of one buffer, and a buffer's size is one page.
                cell_bytes=tuple(
                    buffer.size
                    for layer in grouped_layers
                    for buffer in _fetched_sparse_buffers(layer)
                ),
            )
        )
    return shapes


def sparse_arena_leases(config: KVCacheManagerConfig) -> int:
    """Return the leases each arena gets: the largest constraint's request count."""
    return max((len(batch.kv_caches) for batch in config.constraints), default=0)


def sparse_arena_bytes(config: KVCacheManagerConfig) -> int:
    """Return the GPU bytes of every sparse arena ``config`` makes the manager allocate."""
    max_leases = sparse_arena_leases(config)
    if max_leases <= 0:
        return 0
    shapes = sparse_arena_shapes(config.layers, tokens_per_block=config.tokens_per_block)
    return sum(shape.arena_bytes(max_leases) for shape in shapes)


def sparse_history_bytes_per_block(layers: Sequence[LayerConfig]) -> int:
    """Return the host bytes one request's offloaded history takes per page of tokens.

    Every sparse buffer keeps its history in host-backed pages, and a buffer's
    size is one page, so this is the summed size of the sparse buffers. A
    windowed sparse layer is counted as if it kept its whole history.
    """
    return sum(buffer.size for layer in layers for buffer in layer.buffers if buffer.is_sparse)


@dataclass(frozen=True)
class LifeCyclePage:
    """What one page of a life cycle costs, split by where the native manager puts it.

    Attributes:
        window_size: Sliding window in tokens, or None when the life cycle keeps
            all of its history.
        num_sink_tokens: Tokens kept from the start of the sequence beside the
            window.
        is_state: Whether the life cycle holds one slot of state per request,
            as SSM layers do, rather than pages of tokens.
        device_bytes: Bytes of its dense buffers, which the device quota pays for.
        host_bytes: Bytes of its sparse buffers, which live in pinned host memory.
    """

    window_size: int | None
    num_sink_tokens: int
    is_state: bool
    device_bytes: int
    host_bytes: int

    def pages_per_request(
        self, *, capacity: int, capacity_headroom: int, tokens_per_block: int
    ) -> int:
        """Return the most pages one decoding request holds.

        Args:
            capacity: Tokens the request's cache holds, its history included.
            capacity_headroom: Tokens a decode step reserves above the history.
            tokens_per_block: Tokens per page.
        """
        if self.is_state:
            return 1
        pages = ceil_div(capacity, tokens_per_block)
        if self.window_size is None:
            return pages
        # The live interval spans window + headroom - 1 tokens, at any page offset.
        window_pages = ceil_div(self.window_size + capacity_headroom - 2, tokens_per_block) + 1
        sink_pages = ceil_div(self.num_sink_tokens, tokens_per_block)
        return min(pages, window_pages + sink_pages)


def life_cycle_pages(layers: Sequence[LayerConfig], tokens_per_block: int) -> list[LifeCyclePage]:
    """Return the life cycles ``layers`` form, in the order the native manager numbers them.

    The native manager numbers life cycles by their first layer, which is also
    the order of ``initial_pool_ratio``. Attention layers share a life cycle
    when their window and sink pages match; all SSM layers share one.

    Raises:
        ValueError: A life cycle has sparse buffers but no dense one. The native
            manager sizes a pool group's slots from its device-backed pools, so it
            rejects such a life cycle too.
    """
    keys: list[tuple] = []
    device_bytes: dict[tuple, int] = {}
    host_bytes: dict[tuple, int] = {}
    pages: dict[tuple, tuple[int | None, int, bool]] = {}
    for layer in layers:
        if isinstance(layer, AttentionLayerConfig):
            window = layer.sliding_window_size
            sink_tokens = layer.num_sink_tokens or 0
            key = ("attention", window, ceil_div(sink_tokens, tokens_per_block))
            shape = (window, sink_tokens, False)
        else:
            key = ("state",)
            shape = (None, 0, True)
        if key not in pages:
            keys.append(key)
            pages[key] = shape
            device_bytes[key] = 0
            host_bytes[key] = 0
        for buffer in layer.buffers:
            if buffer.is_sparse:
                host_bytes[key] += buffer.size
            else:
                override = buffer.tokens_per_block_override or tokens_per_block
                device_bytes[key] += buffer.size * (tokens_per_block // override)
    result = []
    for key in keys:
        if host_bytes[key] and not device_bytes[key]:
            raise ValueError(
                "A life cycle with sparse buffers must also hold a dense buffer: the native "
                "KV cache manager sizes a pool group's slots from its device-backed pools."
            )
        window, sink_tokens, is_state = pages[key]
        result.append(
            LifeCyclePage(
                window_size=window,
                num_sink_tokens=sink_tokens,
                is_state=is_state,
                device_bytes=device_bytes[key],
                host_bytes=host_bytes[key],
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class SparseBatchShape:
    """The native Batch the sparse fetch reads, sized once for every step.

    It holds a page table per life cycle, two watermarks and a cache view per row,
    in a device mirror and a pinned host staging copy, and the fetch's work arrays
    on the device. Every step publishes the staging copy to the mirror.

    Attributes:
        num_life_cycles: Life cycles of the manager; each has a page table.
        max_batch_size: Rows.
        max_blocks_per_seq: Columns of each page table.
        selection_widths: Units one step selects, for each life cycle with a sparse
            fetch path.
    """

    num_life_cycles: int
    max_batch_size: int
    max_blocks_per_seq: int
    selection_widths: tuple[int, ...]

    @property
    def mirror_bytes(self) -> int:
        """Device bytes of the mirror: the page tables, watermarks and cache views."""
        entries = self.max_batch_size * (
            self.num_life_cycles * self.max_blocks_per_seq + _BATCH_WATERMARKS_PER_ROW
        )
        views = self.num_life_cycles * self.max_batch_size * _SPARSE_CACHE_VIEW_BYTES
        return pad_up(_INDEX_BYTES * entries, _BATCH_VIEW_ALIGNMENT) + views

    @property
    def staging_bytes(self) -> int:
        """Pinned host bytes of the staging copy of the mirror."""
        return pad_up(self.mirror_bytes, _HOST_PAGE_BYTES)

    @property
    def fetch_scratch_bytes(self) -> int:
        """Device bytes of the fetch's work arrays.

        Per row and life cycle: three entries per selected unit, a miss count and
        the unit in flight.
        """
        return sum(
            _INDEX_BYTES * self.max_batch_size * (3 * width + 1 + _SPARSE_MAX_FLUSH_UNITS)
            for width in self.selection_widths
        )


def sparse_batch_shape(
    layers: Sequence[LayerConfig],
    *,
    tokens_per_block: int,
    max_batch_size: int,
    max_blocks_per_seq: int,
) -> SparseBatchShape:
    """Return the Batch a manager with ``layers`` steps its requests in.

    Args:
        layers: Every layer config the manager allocates.
        tokens_per_block: Tokens per page.
        max_batch_size: Rows of the Batch.
        max_blocks_per_seq: Pages the longest request can hold.
    """
    return SparseBatchShape(
        num_life_cycles=len(life_cycle_pages(layers, tokens_per_block)),
        max_batch_size=max_batch_size,
        max_blocks_per_seq=max_blocks_per_seq,
        selection_widths=tuple(
            grouped_layers[0].sparse_cache.num_selected_blocks
            for grouped_layers in _fetch_path_life_cycles(layers, tokens_per_block)
        ),
    )


# ---------------------------------------------------------------------------
# Sizing policy: the GPU cache per request, the quota split, the host budget and
# the device quota it bounds.
# ---------------------------------------------------------------------------


def sparse_cache_cells(cache_factor: float, num_selected_units: int, units_per_cell: int) -> int:
    """Return the cells one sparse buffer gets in each request's device cache.

    The cache holds ``cache_factor`` selections plus the unit being written,
    rounded up to whole cells. A request has at most one unit in flight, so that
    unit adds one whatever the factor. A factor of 1 is the smallest cache the
    native manager accepts.

    Args:
        cache_factor: Selections the cache holds; at least 1.
        num_selected_units: Units one step selects.
        units_per_cell: Units in one cell, i.e. in one page.
    """
    if cache_factor < 1:
        raise ValueError(f"cache_factor must be at least 1, got {cache_factor}")
    if num_selected_units <= 0 or units_per_cell <= 0:
        raise ValueError(
            "num_selected_units and units_per_cell must be positive, got "
            f"{num_selected_units} and {units_per_cell}"
        )
    # Round before taking the ceiling so float error cannot add a unit.
    selected_units = math.ceil(round(cache_factor * num_selected_units, 6))
    return ceil_div(selected_units + 1, units_per_cell)


def sparse_cache_config(
    *,
    num_selected_units: int,
    unit_tokens: int | None,
    tokens_per_block: int,
    gpu_cache_factor: float,
) -> SparseCacheConfig:
    """Return a layer's sparse fetch path whose cache holds ``gpu_cache_factor`` selections.

    A selection unit is what one selection names: a token, a compressed row or
    pool of tokens, or a whole page. Each request caches selected units in GPU
    cells of one page each. The cache holds ``gpu_cache_factor`` selections plus
    the unit being written, rounded up to whole cells.

    Args:
        num_selected_units: Units one decode step selects for one request. When
            heads select separately, this is the size of their union.
        unit_tokens: Tokens one unit covers, or None when a unit is a whole page.
            Must divide ``tokens_per_block``.
        tokens_per_block: Tokens per page.
        gpu_cache_factor: Selections the cache holds; at least 1.
    """
    if unit_tokens is not None and (unit_tokens <= 0 or tokens_per_block % unit_tokens):
        raise ValueError(
            f"unit_tokens ({unit_tokens}) must divide tokens_per_block ({tokens_per_block})"
        )
    units_per_page = sparse_units_per_page(unit_tokens, tokens_per_block)
    cells = sparse_cache_cells(gpu_cache_factor, num_selected_units, units_per_page)
    return SparseCacheConfig(
        num_selected_blocks=num_selected_units,
        max_cached_units=cells * units_per_page,
        unit_tokens=unit_tokens,
    )


def sparse_offload_pool_ratio(
    life_cycles: Sequence[LifeCyclePage],
    *,
    capacity: int,
    capacity_headroom: int,
    tokens_per_block: int,
) -> list[float]:
    """Return the device quota's split across life cycles for decoding requests.

    Each life cycle's share is the device bytes its pages take in a request of
    ``capacity`` tokens, so the quota buys the same number of such requests in
    every life cycle. Host-backed pools cost the quota nothing and get no share.

    Args:
        life_cycles: The manager's life cycles, in native order.
        capacity: Tokens one request's cache holds.
        capacity_headroom: Tokens a decode step reserves above the history.
        tokens_per_block: Tokens per page.

    Returns:
        One positive share per life cycle, summing to 1; the order and meaning of
        ``KVCacheManagerConfig.initial_pool_ratio``.
    """
    weights = [
        life_cycle.device_bytes
        * life_cycle.pages_per_request(
            capacity=capacity,
            capacity_headroom=capacity_headroom,
            tokens_per_block=tokens_per_block,
        )
        for life_cycle in life_cycles
    ]
    if any(weight <= 0 for weight in weights):
        raise ValueError(f"Every life cycle needs device bytes to share the quota, got {weights}")
    total = sum(weights)
    return [weight / total for weight in weights]


def host_bounded_quota(
    life_cycles: Sequence[LifeCyclePage],
    pool_ratio: Sequence[float],
    *,
    host_budget: int,
    quota: int,
) -> int:
    """Return ``quota``, lowered so the host-backed pools it sizes fit in ``host_budget``.

    A life cycle gets the slots its share of the quota buys at its device bytes
    per slot, and each of those slots also holds its host bytes. A pool group
    can take up to one allocation granule more than its share, so a granule's
    worth of host memory is held back per life cycle with host-backed pages.

    Args:
        life_cycles: The manager's life cycles, in native order.
        pool_ratio: The quota's split across ``life_cycles``.
        host_budget: Host bytes the host-backed pools may take.
        quota: The device quota before the bound.

    Raises:
        ValueError: ``pool_ratio`` does not have one share per life cycle.
    """
    if len(pool_ratio) != len(life_cycles):
        raise ValueError(
            f"The pool ratio has {len(pool_ratio)} shares for {len(life_cycles)} life cycles"
        )
    granule_bytes = gpu_pool_granule_bytes(quota)
    host_per_quota_byte = 0.0
    held_back = 0.0
    for life_cycle, share in zip(life_cycles, pool_ratio):
        if not life_cycle.host_bytes:
            continue
        host_per_device_byte = life_cycle.host_bytes / life_cycle.device_bytes
        host_per_quota_byte += share * host_per_device_byte
        held_back += granule_bytes * host_per_device_byte
    if not host_per_quota_byte:
        return quota
    return min(quota, int(max(0.0, host_budget - held_back) / host_per_quota_byte))


def plan_host_history_bytes(
    *,
    host_cache_size: int | None,
    max_batch_size: int,
    host_bytes_per_request: int,
    pinnable_host_bytes: float,
) -> int:
    """Return the host bytes offloaded history may take.

    An explicit ``host_cache_size`` is the budget. Otherwise the budget holds
    ``max_batch_size`` requests, capped by the host memory the rank may pin.

    Args:
        host_cache_size: The configured size, or None to size automatically.
        max_batch_size: Requests the automatic budget holds.
        host_bytes_per_request: Host history bytes one request holds.
        pinnable_host_bytes: Host bytes automatic sizing may pin; ``inf`` when
            unknown.
    """
    if host_cache_size is not None:
        return host_cache_size
    return int(max(0, min(max_batch_size * host_bytes_per_request, pinnable_host_bytes)))


# ---------------------------------------------------------------------------
# Capacity report and host admission.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseOffloadCapacity:
    """How many requests of one length a worker with sparse KV cache offload holds.

    Attributes:
        num_tokens: Tokens per request.
        max_batch_size: The configured batch limit.
        gpu_pool_bytes: Device bytes of the KV pools.
        host_pool_bytes: Host bytes of the host-backed pools the manager allocated.
        host_budget_bytes: Host bytes offloaded history may take.
        gpu_bytes_per_request: Device pool bytes one request holds.
        fixed_gpu_bytes_per_request: The part of ``gpu_bytes_per_request`` that
            does not grow with length, such as sliding windows.
        host_bytes_per_request: Host history bytes one request holds.
        arena_bytes: Device bytes of the sparse cache arenas.
        num_leases: Leases each arena holds.
        lease_bytes: Device bytes of one lease, summed over the arenas.
        batch_buffer_bytes: Device bytes of the buffers sized for
            ``max_batch_size`` rows: the Batch and the model's offload scratch.
        by_limit: Requests that fit by each limit: ``max_batch_size``, ``host``
            and ``gpu``, in the order that breaks ties.
        recommended_max_batch_size: The largest batch the device budget and the
            host budget hold.
    """

    num_tokens: int
    max_batch_size: int
    gpu_pool_bytes: int
    host_pool_bytes: int
    host_budget_bytes: int
    gpu_bytes_per_request: int
    fixed_gpu_bytes_per_request: int
    host_bytes_per_request: int
    arena_bytes: int
    num_leases: int
    lease_bytes: int
    batch_buffer_bytes: int
    by_limit: dict[str, int] = field(compare=False)
    recommended_max_batch_size: int

    @property
    def limited_by(self) -> str:
        """The limit that admits the fewest requests."""
        return min(self.by_limit, key=self.by_limit.__getitem__)

    @property
    def fits(self) -> int:
        """Requests that fit."""
        return self.by_limit[self.limited_by]

    def describe(self) -> str:
        """Return a one-line report for the log."""
        return (
            f"Sparse KV cache offload capacity at {self.num_tokens} tokens per request: "
            f"GPU pools {self.gpu_pool_bytes / _GIB:.2f}GiB at "
            f"{self.gpu_bytes_per_request / _MIB:.1f}MiB per request "
            f"({self.fixed_gpu_bytes_per_request / _MIB:.1f}MiB fixed); arena "
            f"{self.arena_bytes / _GIB:.2f}GiB for {self.num_leases} leases of "
            f"{self.lease_bytes / _MIB:.2f}MiB; offload buffers "
            f"{self.batch_buffer_bytes / _MIB:.1f}MiB for {self.max_batch_size} rows; "
            f"host-backed history pools "
            f"{self.host_pool_bytes / _GIB:.2f}GiB at "
            f"{self.host_bytes_per_request / _MIB:.1f}MiB per request, host budget "
            f"{self.host_budget_bytes / _GIB:.2f}GiB. Requests: {self.by_limit['gpu']} by "
            f"GPU, {self.by_limit['host']} by host, max_batch_size {self.max_batch_size}; "
            f"{self.fits} fit, limited by {self.limited_by}. A max_batch_size of "
            f"{self.recommended_max_batch_size} fits the GPU budget and the host budget."
        )


def solve_sparse_offload_capacity(
    *,
    num_tokens: int,
    max_batch_size: int,
    max_admissible_sequences: int,
    gpu_pool_bytes: int,
    host_pool_bytes: int,
    host_budget_bytes: int,
    gpu_bytes_per_request: int,
    fixed_gpu_bytes_per_request: int,
    host_bytes_per_request: int,
    arenas: Sequence[SparseArenaShape],
    num_leases: int,
    batch_buffer_bytes: int,
) -> SparseOffloadCapacity:
    """Return how many requests of ``num_tokens`` tokens fit, and the batch size that fits.

    Args:
        num_tokens: Tokens per request.
        max_batch_size: The configured batch limit.
        max_admissible_sequences: Requests that hold a lease at once, excluding
            padding requests: a batch slot holds two when a retiring or
            receiving request keeps its lease beside the next one.
        gpu_pool_bytes: Device bytes of the KV pools.
        host_pool_bytes: Host bytes of the host-backed pools the manager allocated.
        host_budget_bytes: Host bytes offloaded history may take.
        gpu_bytes_per_request: Device pool bytes one request holds.
        fixed_gpu_bytes_per_request: The part of ``gpu_bytes_per_request`` that
            does not grow with length.
        host_bytes_per_request: Host history bytes one request holds.
        arenas: The arena of each sparse life cycle.
        num_leases: Leases each arena holds.
        batch_buffer_bytes: Device bytes of the buffers sized for
            ``max_batch_size`` rows, reserved apart from ``gpu_pool_bytes``.
    """
    gpu_bytes_per_request = max(1, gpu_bytes_per_request)
    host_bytes_per_request = max(1, host_bytes_per_request)
    arena_bytes = sum(arena.arena_bytes(num_leases) for arena in arenas)
    lease_bytes = sum(arena.lease_bytes for arena in arenas)
    by_limit = {
        "max_batch_size": max_batch_size,
        "host": host_pool_bytes // host_bytes_per_request,
        "gpu": gpu_pool_bytes // gpu_bytes_per_request,
    }
    # The device budget pays for the pools, the arena and the buffers sized for
    # max_batch_size rows. The arena's shared tables and padding leases are
    # fixed; every batch slot adds a request's pages, the leases that slot holds
    # and its rows of those buffers.
    leases_per_slot = max(1, max_admissible_sequences // max(1, max_batch_size))
    padding_leases = max(0, num_leases - max_admissible_sequences)
    fixed_arena_bytes = sum(arena.arena_bytes(padding_leases) for arena in arenas)
    batch_bytes_per_slot = ceil_div(batch_buffer_bytes, max(1, max_batch_size))
    recommended = min(
        max(0, gpu_pool_bytes + arena_bytes + batch_buffer_bytes - fixed_arena_bytes)
        // (gpu_bytes_per_request + leases_per_slot * lease_bytes + batch_bytes_per_slot),
        host_budget_bytes // host_bytes_per_request,
    )
    return SparseOffloadCapacity(
        num_tokens=num_tokens,
        max_batch_size=max_batch_size,
        gpu_pool_bytes=gpu_pool_bytes,
        host_pool_bytes=host_pool_bytes,
        host_budget_bytes=host_budget_bytes,
        gpu_bytes_per_request=gpu_bytes_per_request,
        fixed_gpu_bytes_per_request=fixed_gpu_bytes_per_request,
        host_bytes_per_request=host_bytes_per_request,
        arena_bytes=arena_bytes,
        num_leases=num_leases,
        lease_bytes=lease_bytes,
        batch_buffer_bytes=batch_buffer_bytes,
        by_limit=by_limit,
        recommended_max_batch_size=recommended,
    )


class HostHistoryReservations:
    """Host history pages reserved for admitted requests through their last token.

    A request whose history outgrows the host-backed pool while it decodes has
    nowhere to go, so admission reserves every page a request can reach.
    """

    def __init__(self, capacity_blocks: int) -> None:
        """Track reservations against a pool of ``capacity_blocks`` pages."""
        self.capacity_blocks = capacity_blocks
        self.reserved_blocks = 0
        self._blocks_by_request: dict[int, int] = {}

    def can_reserve(self, request_id: int, num_blocks: int) -> bool:
        """Return whether ``num_blocks`` pages fit for ``request_id``, replacing its reservation."""
        held = self._blocks_by_request.get(request_id, 0)
        return self.reserved_blocks - held + num_blocks <= self.capacity_blocks

    def reserve(self, request_id: int, num_blocks: int) -> None:
        """Reserve ``num_blocks`` pages for ``request_id``, replacing its reservation."""
        held = self._blocks_by_request.get(request_id, 0)
        self._blocks_by_request[request_id] = num_blocks
        self.reserved_blocks += num_blocks - held

    def release(self, request_id: int) -> None:
        """Release the pages reserved for ``request_id``, if any."""
        self.reserved_blocks -= self._blocks_by_request.pop(request_id, 0)
