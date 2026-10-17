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

import array
import enum
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Iterator,
    NamedTuple,
    NewType,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    Union,
    overload,
)

# From _common.py
NDEBUG: Final[int]
DEFAULT_BEAM_INDEX: Final[BeamIndex]
BAD_PAGE_INDEX: Final[int]
GPU_LEVEL: Final[CacheLevel]
CACHE_LEVEL1: Final[CacheLevel]

class CorruptedError(Exception):
    """Raised by every public entry point once a broken invariant has been recorded."""

class CuError(Exception):
    """A CUDA driver call failed; carries the driver's own status code."""

    error_code: Any

class OutOfMemoryError(Exception): ...
class OutOfPagesError(OutOfMemoryError): ...

def poison_reason() -> str | None:
    """First recorded invariant violation, or None. Never clears, so it is safe to poll."""

def take_poison() -> str | None:
    """Report the recorded violation and clear it, but only once no manager is alive."""

def num_live_managers() -> int:
    """Number of constructed, not-yet-destroyed managers."""

class CacheTier(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2

class PageStatus(enum.Enum):
    LOCKED = enum.auto()
    HELD = enum.auto()
    DROPPABLE = enum.auto()

class PageIndexMode(enum.IntEnum):
    SHARED = 0
    PER_LAYER = 1

LifeCycleId = NewType("LifeCycleId", int)
LayerGroupId: TypeAlias = LifeCycleId

class AttnLifeCycle:
    """The attention life cycle, keyed by its window and sink-token shape.

    Not by sparsity: that is a property of a pool, so one life cycle may hold both sparse
    and dense buffers and two layers differing only in sparsity share one life cycle.
    """

    @staticmethod
    def make(
        window_size: int | None,
        num_sink_tokens: int | None,
        tokens_per_block: int,
    ) -> "AttnLifeCycle": ...
    @property
    def window_size(self) -> int | None: ...
    @property
    def num_sink_blocks(self) -> int: ...
    def get_stale_range(self, history_length: int, tokens_per_block: int) -> HalfOpenRange: ...

CacheLevel = NewType("CacheLevel", int)
TokenId = NewType("TokenId", int)
TokenIdExt = Union[TokenId, bytes]

class PlannedDropHandle:
    def drop(self) -> None: ...

class ReuseScope(NamedTuple):
    lora_id: int | None = None
    salt: int | None = None

SlidingWindowSize: TypeAlias = int | None
LayerId = NewType("LayerId", int)
CudaStream = NewType("CudaStream", int)
BeamIndex = NewType("BeamIndex", int)
MemAddress = NewType("MemAddress", int)
Priority = NewType("Priority", int)
PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)

# From _storage_manager.py
class StorageStatistics:
    """Independent per-pool storage counts returned by the manager."""

    @property
    def slot_sizes(self) -> list[int]: ...
    @property
    def total(self) -> int: ...
    @property
    def free(self) -> int: ...
    @property
    def evictable(self) -> int: ...
    @property
    def available(self) -> int: ...
    @property
    def unavailable(self) -> int: ...

# From _stats.py
@dataclass(slots=True)
class KVCacheStatsDelta:
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0

@dataclass(slots=True)
class KVCacheIterationStatsDelta:
    iter_alloc_total_blocks: int = 0
    iter_alloc_new_blocks: int = 0
    iter_reused_blocks: int = 0
    iter_full_reused_blocks: int = 0
    iter_partial_reused_blocks: int = 0
    iter_missed_blocks: int = 0
    iter_gen_alloc_blocks: int = 0
    iter_onboard_blocks: int = 0
    iter_onboard_bytes: int = 0
    iter_offload_blocks: int = 0
    iter_offload_bytes: int = 0
    iter_intra_device_copy_blocks: int = 0
    iter_intra_device_copy_bytes: int = 0
    iter_host_dropped_blocks: int = 0
    iter_host_dropped_bytes: int = 0

@dataclass(slots=True)
class ReusedBlocksByLevel:
    """Reuse block counts split by the cache level the reused pages were resident on.

    Indices are CacheLevel values, so entry i is the i-th configured tier.
    """

    full: list[int]
    partial: list[int]

class SsmSnapshotIterationStatsDelta:
    iter_snapshot_lookups: int = 0
    iter_snapshot_hits: int = 0
    iter_snapshot_misses: int = 0
    iter_reused_tokens: int = 0
    iter_unreused_tokens: int = 0
    iter_aligned_snapshot_hits: int = 0
    iter_unaligned_snapshot_hits: int = 0
    @property
    def iter_snapshot_hit_rate(self) -> float: ...

@dataclass(slots=True, frozen=True)
class PoolGroupPeakBlockStats:
    available: int
    unavailable: int
    evictable: int

class DeviceArray(Protocol):
    """A dense array in GPU memory, borrowed for the duration of a call.

    Structural, not nominal: anything exporting the DLPack protocol over a CUDA buffer
    satisfies it, so a ``torch.Tensor`` is accepted without this package depending on
    torch. Callers may equally pass a CuPy, JAX or numba device array.

    Shape and dtype are stated per parameter rather than in the type, and are checked at
    the boundary. Contents are read or written on the caller's stream; the array must
    stay alive and unmodified until that work completes.
    """

    def __dlpack__(self, *, stream: int | None = ...) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

# From _config.py
DataRole = NewType("DataRole", str)

class CacheTierConfig(Protocol):
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class GpuCacheTierConfig:
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class HostCacheTierConfig:
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True)
class DiskCacheTierConfig:
    quota: int
    path: str
    @property
    def tier(self) -> CacheTier: ...
    def assert_valid(self) -> None: ...

@dataclass(slots=True, frozen=True)
class BufferConfig:
    """One buffer of a layer's KV cache.

    ``is_sparse`` marks a buffer whose history is too large for device memory and is read
    through a per-request device cache rather than directly (see
    :meth:`KVCacheManager.is_sparse`).

    It changes the *medium* its pages are made of, not where they lock. Every buffer locks
    its pages at the hot level; a sparse buffer's life cycle gets a host-backed hot pool
    group, device-mapped so a kernel can read it. Its pages are otherwise ordinary hot
    pages -- coalesced, addressed by page index, visible to the transceiver -- and the
    page-index table stays uniform. A sparse buffer places no requirement on the cold
    tiers, which may be host, disk, or absent.

    Backing is decided per pool group, and pool groups are formed per lifecycle, so a
    sparse buffer requires a lifecycle of its own; sharing one with a non-sparse buffer
    raises from the ``KVCacheManager`` constructor.

    All device residency is a per-request lease on the life cycle's cache arena -- a
    dedicated allocation, not a storage tier. Both the unit being written and the history a
    step selects are cells in it. ``max_cached_units`` is the whole cache, counting both
    populations, and the lease follows from it alone. The two never overlap: the selection
    is below the watermark, everything in flight is above it. Only one unit is ever in
    flight, because a request contributes one query token per step, so the surplus above the
    selection width is retention rather than headroom.

    A lease is one contiguous run of buffer-sized cells, so an entry is a
    ``(buffer, entry index)`` pair resolved to a cell by arithmetic alone -- no page table
    and no slot indirection. :meth:`KVCacheManager.fetch_sparse_pages` fills the cells a
    step selects.

    Two to four times ``num_selected_blocks`` is the usual starting point: a cache only as
    wide as the selection retains nothing between steps, and retention of units not selected
    this step is where the hit rate comes from. ``max_cached_units`` must exceed
    ``num_selected_blocks``, since equality would leave no cell for the unit being written.

    Both default to zero, meaning the fetch path is not configured for this buffer; sparse
    storage and locking work without them.

    ``sparse_unit_layout`` tells the manager where a selection unit's bytes are. It is
    required once ``max_cached_units`` is set, because a cell that cannot be located can be
    neither filled nor written back.

    A block's data reaches its page by a write-back that leaves the cell cached and
    merely stops pinning it. A block must be written back before it can be selected, so
    :meth:`_KVCache.resize` is both the deadline and the only place it happens -- that being
    the one call through which the watermark ever moves. ``commit()`` is not a separate
    write-back site: a committed block is already at or below the watermark, and where
    ``commit()`` advances the watermark it calls :meth:`_KVCache.resize`. A cell is
    evictable exactly when its unit is below the watermark, which is what protects the block
    still being written.

    Fixed at construction: the storage layout derived from it is built once.
    """

    role: DataRole
    size: int
    tokens_per_block_override: int | None = None
    is_sparse: bool = False

    sparse_unit_chunks: int = 1
    """Pieces a selection unit is gathered from within *this buffer's* page.

    Per buffer, because it is this buffer's own dimension order:

    - **1** for a sequence-major page (``[token][head][dim]``), where a unit's tokens carry
      their heads with them and the whole unit is one contiguous range. Also 1 for any
      single-head page -- a compressed or pooled latent -- where the orders coincide.
    - **numHeads** for a head-major page (``[head][token][dim]``), where each head
      contributes its own run and a unit is that many strided pieces.

    Every sparse model in this repository today is one of the first two cases, so 1 is
    almost always right. The rest of the layout follows from this, the layer's
    ``sparse_cache.unit_tokens`` and the page being densely packed, so a padded page is not
    expressible.
    """

@dataclass(slots=True)
class SparseCacheConfig:
    """One layer's sparse fetch path.

    Per layer rather than per buffer because every sparse buffer of a layer shares one
    resolve, one resident set and one recency order: the metadata is laid out
    ``[layer][entry]``, and a selection names units for all of them at once. Declaring
    these per buffer would let two of them disagree with nothing to reconcile it.

    Its presence is what configures the fetch path. A sparse buffer in a layer without it
    still gets sparse storage and locking; it simply has no GPU cache to read through.

    Layers sharing a life cycle must declare the same values -- they share one arena.
    """

    num_selected_blocks: int
    """Selection width: units a sparse-attention step selects for this layer."""

    max_cached_units: int
    """Cells one buffer gets in a request's GPU cache, counting both populations: the
    history it caches and every unit still in flight.

    Two to four times ``num_selected_blocks`` is the usual starting point: a cache only as
    wide as the selection retains nothing between steps, and retention of units not
    selected this step is where the hit rate comes from. It must exceed
    ``num_selected_blocks``, since equality would leave no cell for the unit being written.
    """

    unit_tokens: int | None = None
    """Original tokens one selection unit spans. ``None`` means a unit is a whole page.

    Must divide ``tokens_per_block``; the quotient is how many units a page holds.
    Distinct from ``BufferConfig.tokens_per_block_override``, which reinterprets a buffer's
    page for *storage*: this names the sparse cache's granularity and nothing else, so
    pages, page indices and index spaces all keep the manager's ``tokens_per_block``.
    Setting both on one buffer is rejected.
    """

@dataclass(slots=True)
class AttentionLayerConfig:
    """One attention layer's buffers and its history rules.

    A sparse layer may also declare a sliding window, though the combination is unusual:
    sparsity already bounds how much history a step reads. Where both apply, the window
    bounds what is *selectable* as well as what is retained -- the indexer producing the
    selection applies the same window, so it never names a block the window has made stale.
    """

    layer_id: LayerId
    buffers: list[BufferConfig]
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None

    sparse_cache: SparseCacheConfig | None = None
    """The sparse fetch path for this layer's sparse buffers, if it has one.

    Setting it on a layer with no sparse buffer to serve is rejected.
    """
    @property
    def window_size(self) -> int | None: ...

@dataclass(slots=True)
class SsmLayerConfig:
    layer_id: LayerId
    buffers: list[BufferConfig]

LayerConfig = AttentionLayerConfig | SsmLayerConfig

@dataclass(slots=True)
class KVCacheDesc:
    capacity: int
    history_length: int
    beam_width: int = 1
    prompt_length: int = 0

@dataclass(slots=True)
class BatchDesc:
    kv_caches: list[KVCacheDesc]
    system_prompt_length: int = 0

@dataclass(slots=True)
class SwaScratchReuseConfig:
    max_rewind_len: int = 0

@dataclass(slots=True)
class KVCacheManagerConfig:
    tokens_per_block: int
    cache_tiers: list[CacheTierConfig]
    layers: list[LayerConfig]
    max_util_for_resume: float = ...
    enable_partial_reuse: bool = True
    # Tokens trimmed off the tail of every prefix match; nonzero only for a pool
    # that also holds state reading that many tokens ahead of each position.
    reuse_match_backoff: int = 0
    constraints: list[BatchDesc] = ...
    typical_step: BatchDesc | None = None
    # One positive, normalized hot-tier byte-quota weight per layer group. Cold initialization preserves the implied
    # layer-group slot-count proportions.
    initial_pool_ratio: list[float] | None = None
    swa_scratch_reuse: SwaScratchReuseConfig | None = None
    commit_min_snapshot: bool = False
    enable_stats: bool = True
    text_only: bool = False
    enable_partial_commit: bool = True
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...

# From _event_manager.py
EventBlockHash: TypeAlias = int | str
BlockHashLike: TypeAlias = bytes | EventBlockHash
BlockHashesLike: TypeAlias = BlockHashLike | Iterable[BlockHashLike]
EventTokenId: TypeAlias = int | str
MmKey: TypeAlias = tuple[bytes, int] | tuple[bytes, int, str | None]
AttentionDpGatherFn: TypeAlias = Callable[[list["KVCacheEvent"]], list[list["KVCacheEvent"]]]

@dataclass(slots=True, frozen=True)
class UniqueToken:
    token_id: EventTokenId
    token_extra_id: int = ...

@dataclass(slots=True, frozen=True)
class KVCacheCreatedData:
    num_blocks_per_cache_level: list[int]

@dataclass(slots=True, frozen=True)
class KVCacheStoredBlockData:
    block_hash: EventBlockHash
    tokens: list[UniqueToken]
    cache_level: int
    priority: int
    mm_keys: list[MmKey] = ...
    cache_salt: str | None = ...

@dataclass(slots=True, frozen=True)
class KVCacheStoredData:
    parent_hash: EventBlockHash | None
    blocks: list[KVCacheStoredBlockData]

@dataclass(slots=True, frozen=True)
class KVCacheRemovedData:
    block_hashes: list[EventBlockHash]

@dataclass(slots=True, frozen=True)
class KVCacheEventDiff:
    old_value: int
    new_value: int

@dataclass(slots=True, frozen=True)
class KVCacheUpdatedData:
    block_hash: EventBlockHash
    cache_level: KVCacheEventDiff | None
    priority: KVCacheEventDiff | None

@dataclass(slots=True, frozen=True)
class KVCacheEvent:
    event_id: int
    data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData
    window_size: int
    hash_algo: str | None = None
    attention_dp_rank: int | None = None
    layer_group_id: int | None = None

class StreamingBlockStoredData:
    @property
    def lora_id(self) -> int | None: ...
    @property
    def block_hashes(self) -> list[int]: ...
    @property
    def parent_block_hash(self) -> int | None: ...
    @property
    def token_ids(self) -> list[EventTokenId]: ...
    @property
    def mm_keys(self) -> list[list[MmKey]]: ...

class StreamingBlockRemovedData:
    @property
    def block_hashes(self) -> list[int]: ...

class StreamingEventStats:
    @property
    def stored_blocks(self) -> int: ...
    @property
    def removed_blocks(self) -> int: ...
    @property
    def partial_blocks_suppressed(self) -> int: ...
    @property
    def non_target_life_cycles_ignored(self) -> int: ...
    @property
    def dropped_events(self) -> int: ...

class StreamingEventSink:
    def __init__(
        self,
        max_entries: int = ...,
        mm_token_id_offset: int | None = None,
    ) -> None: ...
    def set_target_life_cycle(self, life_cycle_id: int) -> None: ...
    def drain_iteration_events(
        self,
    ) -> list[StreamingBlockStoredData | StreamingBlockRemovedData]: ...
    @property
    def stats(self) -> StreamingEventStats: ...

class KVCacheEventManager:
    def __init__(
        self,
        max_kv_event_entries: int,
        *,
        window_size: int = ...,
        attention_dp_rank: int | None = None,
        attention_dp_gather: AttentionDpGatherFn | None = None,
        hash_algo: str = ...,
        window_size_by_layer_group: dict[int, int] | None = None,
        mm_token_id_offset: int | None = None,
    ) -> None: ...
    def add_created_event(
        self,
        num_blocks_per_cache_level: Sequence[int],
        layer_group_ids: Sequence[int] | None = None,
    ) -> None: ...
    def set_layer_group_window_sizes(self, window_sizes: dict[int, int]) -> None: ...
    def add_stored_event(
        self,
        parent_hash: EventBlockHash | None,
        blocks: Sequence[KVCacheStoredBlockData],
        layer_group_id: int | None = None,
    ) -> None: ...
    def add_stored_block_event_from_block(self, block: Any) -> None: ...
    def add_stored_life_cycle_event_from_block(self, block: Any, life_cycle_id: int) -> None: ...
    def add_removed_event(self, block_hashes: BlockHashesLike) -> None: ...
    def add_removed_life_cycle_event(self, block_hash: bytes, life_cycle_id: int) -> None: ...
    def add_updated_event(
        self,
        block_hash: BlockHashLike,
        *,
        cache_level: KVCacheEventDiff | None = None,
        priority: KVCacheEventDiff | None = None,
        layer_group_id: int | None = None,
    ) -> None: ...
    def flush_iteration_events(self) -> None: ...
    def get_latest_events(self, timeout_ms: float | None = None) -> list[KVCacheEvent]: ...

# Native key builders, shared with the radix tree so routing hashes match the engine's.
def gen_multimodal_cache_key_tokens(
    id_offset: int,
    multi_modal_data_digest: bytes,
    num_tokens: int,
    token_offset: int = 0,
) -> list[TokenIdExt]: ...
def sequence_to_blockchain_keys(
    tokens_per_block: int,
    reuse_scope: ReuseScope,
    tokens: Sequence[TokenIdExt],
) -> Iterator[tuple[list[TokenIdExt], bytes]]: ...

# From _core/_kv_cache.py
class _Status(enum.Enum):
    ACTIVE = enum.auto()
    SUSPENDED = enum.auto()
    CLOSED = enum.auto()

KvCacheStatus: TypeAlias = _Status

IndexSeq = array.array[int] | memoryview[int]

class _KVCache:
    Status: ClassVar[Type[_Status]]
    id: Any
    def __init__(
        self,
        manager: "KVCacheManager",
        reuse_scope: ReuseScope,
        reuse_match: Any | None,
        id: Any,
        custom_priority_callback: Callable[[int, Any], Priority],
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
    ) -> None: ...
    def set_base_page_index_buf(
        self, beam_idx: BeamIndex, layer_group_id: LayerGroupId, buf: memoryview | None
    ) -> None:
        """Write page indices through ``buf`` instead of this cache's own storage.

        Raises while this cache belongs to a :class:`Batch`: membership already binds every
        layer group to a row of the batch's table, and a second buffer would take the writes
        with it, leaving that row holding whatever it held when the cache joined. Remove it
        from the batch first, or give the batch the buffer worth writing to.
        """
    @property
    def manager(self) -> "KVCacheManager": ...
    @property
    def cuda_stream(self) -> CudaStream: ...
    @cuda_stream.setter
    def cuda_stream(self, cuda_stream: CudaStream) -> None: ...
    @property
    def num_blocks(self) -> int: ...
    @property
    def batch_row(self) -> int:
        """Row this cache occupies in its :class:`Batch`, or ``-1`` when it belongs to none.

        Also the answer to who owns its page index buffers: a member's are the batch's
        rows, so :meth:`set_base_page_index_buf` refuses and ``suspend()`` leaves them
        bound rather than copying them back out.
        """
    def commit_pending_stats(self) -> KVCacheStatsDelta: ...
    def discard_pending_stats(self) -> None: ...
    def close(self) -> None: ...
    @property
    def beam_width(self) -> BeamIndex: ...
    @beam_width.setter
    def beam_width(self, beam_width: BeamIndex) -> None:
        """Expand before the first generation step, never during generation (C++ only).

        First resume the cache and materialize prompt storage (or prepare synthetic
        warmup state). Full prompt blocks are shared; the writable tail, including
        preallocated blocks, is copied using the boundary set by
        ``expected_prompt_length`` at cache creation.
        """
        ...
    def get_base_page_indices(
        self, layer_group_id: LayerGroupId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> IndexSeq: ...
    def get_aggregated_page_indices(
        self,
        layer_group_id: LayerGroupId,
        beam_id: BeamIndex = DEFAULT_BEAM_INDEX,
        valid_only: bool = False,
    ) -> Iterator[int]: ...
    def resize(self, capacity: int | None, history_length: int | None = None) -> bool: ...
    @property
    def max_input_length(self) -> int | None:
        """Bound on ``capacity - history_length``: how far capacity may outrun the watermark.

        ``None`` unless a sparse buffer's fetch path is configured, since a bound on
        in-flight blocks describes nothing without a cache to hold them. :meth:`resize`
        rejects a gap wider than this.

        **One token**, and read-only, because it is a consequence of the supported shape
        rather than of how wide a cache is. A sparse request contributes exactly one query
        token per step: :meth:`KVCacheManager.fetch_sparse_pages` resolves one selection
        per request, while an attention kernel indexes its selection per query token, and
        the two agree only at one token each. Prefill and speculative decoding give a
        request several query tokens per step and are out of scope; this bound is the guard
        that keeps them out.

        It follows that at most one unit is ever in flight, so essentially all of
        :attr:`BufferConfig.max_cached_units` is available to the selection rather than
        being reserved as headroom.
        """

    @property
    def capacity(self) -> int: ...
    @capacity.setter
    def capacity(self, capacity: int) -> None: ...
    @property
    def history_length(self) -> int: ...
    @history_length.setter
    def history_length(self, history_length: int) -> None: ...
    def commit(
        self,
        accepted_input_tokens: Sequence[TokenIdExt],
        beam_search_indices: Sequence[int] | None = None,
        is_end: bool = False,
    ) -> None: ...
    @property
    def num_committed_tokens(self) -> int: ...
    @property
    def cached_tokens_by_level(self) -> list[int]: ...
    def _get_last_cached_token_level(self) -> int | None: ...
    @property
    def committed_tokens(self) -> list[TokenIdExt]: ...
    @property
    def reuse_scope(self) -> ReuseScope: ...
    def plan_committed_block_drop(self) -> PlannedDropHandle | None: ...
    def stop_committing(self) -> None: ...
    def suspend(self) -> None: ...
    def resume(self, cuda_stream: CudaStream | None = None) -> bool: ...
    def prefetch(self, target: CacheLevel) -> bool: ...
    def get_scratch_desc(self, layer_group_id: LayerGroupId) -> ScratchDesc | None: ...
    @property
    def has_scratch_slots(self) -> bool: ...
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...
    @enable_swa_scratch_reuse.setter
    def enable_swa_scratch_reuse(self, enable: bool) -> None: ...
    @property
    def text_only(self) -> bool: ...
    @text_only.setter
    def text_only(self, text_only: bool) -> None: ...
    # Per life cycle, which is where the scratch slots live. The one-argument form
    # collapses them, so one life cycle's scratch makes it answer False for every other.
    @overload
    def supports_index_mode(self, mode: PageIndexMode) -> bool: ...
    @overload
    def supports_index_mode(self, mode: PageIndexMode, layer_group_id: LayerGroupId) -> bool: ...
    @property
    def status(self) -> _Status: ...
    @property
    def is_active(self) -> bool: ...
    @property
    def tokens_per_block(self) -> int: ...

@dataclass(slots=True, frozen=True)
class PoolDesc:
    pool_index: PoolIndex
    base_address: MemAddress
    slot_bytes: int

class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole

@dataclass(slots=True, frozen=True)
class ExpandedBuffer:
    id: BufferId
    expansion: int  # expansion factor of page due to heterogeneous tokens_per_block

@dataclass(slots=True, frozen=True)
class AggregatedPageDesc:
    """The data you need would be in the following byte ranges.

    (base + stride * i + Range(0, size) for i in aggregated_page_indices)
    """

    base: MemAddress
    size: int
    stride: int
    layer_group_id: LayerGroupId
    buffers: Sequence[ExpandedBuffer]

@dataclass(slots=True, frozen=True)
class CoalescedBuffer:
    single_buffer_size: int
    buffer_ids: Sequence[BufferId]
    @property
    def size(self) -> int: ...
    @property
    def num_buffers(self) -> int: ...

@dataclass(slots=True, frozen=True)
class SlotDescVariant:
    coalesced_buffers: Sequence[CoalescedBuffer]
    @property
    def layer_group_id(self) -> LayerGroupId: ...
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class SlotDesc:
    variants: Sequence[SlotDescVariant]
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class PoolGroupDesc:
    pool_group_index: PoolGroupIndex
    num_slots: int
    slot_desc: SlotDesc
    pools: Sequence[PoolDesc]

# From _core/_kv_cache_manager.py
class HalfOpenRange:
    def __init__(self, beg: int, end: int) -> None: ...
    @property
    def beg(self) -> int: ...
    @property
    def end(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

@dataclass(slots=True, frozen=True)
class ScratchDesc:
    range: HalfOpenRange
    slot_ids: Sequence[int]
    def __bool__(self) -> bool: ...

@dataclass(slots=True, frozen=True)
class PageIndexConverter:
    scale: int
    expansion: int
    layer_offset: int

    def __call__(
        self,
        base_indices: Sequence[int],
        index_mode: PageIndexMode | None = None,
        scratch: ScratchDesc | None = None,
    ) -> list[int]: ...

class IKvCacheColdPageCodec: ...

def create_default_kv_cache_cold_page_codec() -> IKvCacheColdPageCodec:
    """Create the default lossless cold-page codec.

    Passing ``cold_page_codec=None`` to ``KVCacheManager`` already selects this codec, so normal users do not need to
    call this factory. It is primarily provided to demonstrate how a native codec factory exposes an owning
    ``IKvCacheColdPageCodec`` object for transfer into ``KVCacheManager``. Any ``KVCacheManager`` construction attempt
    consumes an explicitly supplied codec, including an attempt that fails.
    """

class ExternalPageTable:
    """Host page-index storage a :class:`Batch`'s members write into.

    The runtime already holds one of these -- the host block-offsets tensor the dense
    attention path fills -- and its rows carry exactly the numbers a member would otherwise
    write somewhere else as well. A ``_KVCache`` can write to only one buffer at a time, so
    two of them means handing the binding back and forth every step. Pointing the member at
    the runtime's row instead leaves one buffer with one writer, and the binding survives
    the whole decode.

    ``host`` addresses an ``int32`` array laid out as ``[layer_group][row][block]`` with the
    given element strides. A row is longer than ``max_blocks_per_seq`` when the table is a
    plane of a wider array, which is the usual case: the runtime's tensor carries a
    key/value dimension the ``Batch`` does not use.

    Only the host side is lent. The device table stays the ``Batch``'s own, because
    :meth:`Batch.publish` has to produce a snapshot: without one the host could not write a
    row until every kernel reading it had retired, which for a captured graph means the end
    of the forward. The lender's own device mirror is no substitute either -- no captured
    kernel reads it today, which is what lets its upload overlap the previous step, and
    sharing it would give that up.

    The lender owns the rows. A row no member occupies is left as the lender had it,
    because it is serving one of the lender's other requests; and a member must be removed
    from the ``Batch`` before the lender reassigns the row it sits in.
    """

    def __init__(
        self,
        host: int,
        layer_group_stride: int,
        row_stride: int,
    ) -> None: ...
    @property
    def layer_group_stride(self) -> int: ...
    @property
    def row_stride(self) -> int: ...

class IndexTableRequest:
    """Buffers a consumer must address from one table, under one base pointer.

    States a kernel's requirement, not a preference: some attention ops take a single block
    table covering K and V rather than one each. Order defines the table's buffer dimension,
    so a kernel reads buffer ``i`` of this request at entry ``i``.

    Every buffer must share a life cycle and a page-index scale, since no one table can
    serve buffers whose converted indices differ. A request spanning either is refused when
    the ``Batch`` is constructed.
    """

    def __init__(self, buffers: Sequence[BufferId]) -> None: ...
    @property
    def buffers(self) -> list[BufferId]: ...

class Batch:
    """The ``_KVCache`` objects stepped together, and their device-side mirror.

    Owns the batch row order and every per-row device array the sparse path reads: a page
    table per layer group, the eligible-block counts, and the staging buffers behind
    them. Every such array is indexed by row, and each must agree on which row a request
    occupies, so the order lives here rather than being re-established per call.

    Rows are slots rather than positions in a compacted list: removing a member leaves a
    hole for reuse instead of shifting later rows, keeping array shapes and addresses
    stable across steps, which is what CUDA graph capture requires.

    Membership is exclusive -- a ``_KVCache`` belongs to at most one ``Batch``.

    :meth:`publish` restages every row. Mutating a member directly therefore costs a
    republish rather than producing a stale table; refreshing only the rows whose locks
    changed is a planned refinement, and a cost rather than a correctness question.
    """

    def __init__(
        self,
        manager: KVCacheManager,
        max_batch_size: int = 0,
        max_blocks_per_seq: int = 0,
        page_table: ExternalPageTable | None = None,
        mode: PageIndexMode | None = None,
        index_tables: Sequence[IndexTableRequest] = (),
    ) -> None:
        """Bound the device arrays, whose shape must not change between steps.

        Neither can be grown later, so neither is inferred: zero derives it from the
        manager's ``constraints``, which fails when none were declared. A pool-sizing hint
        is a default, never a substitute for a batch bound.

        ``page_table`` borrows the page-index storage instead of allocating it, which
        leaves one buffer with one writer and takes the page indices out of
        :meth:`publish`. Its rows then belong to the lender, so :meth:`add_kv_cache` has to
        be told which one a member occupies.

        ``mode`` applies to every life cycle. Left unset, each takes the smaller table:
        ``SHARED`` where the manager reports it always available, ``PER_LAYER`` otherwise.
        It is fixed here because a converted table's shape is baked into a captured graph,
        so it cannot follow a member's scratch state.
        """

    def add_kv_cache(self, kv_cache: _KVCache, row: int = -1) -> int:
        """Seat a cache in ``row``, or in the lowest free row when it is negative.

        Returns the row. Raises if the row is taken or out of range, if the batch is full,
        if the cache already belongs to a batch, or if its beam width exceeds one -- only
        beam 0 is mirrored, so a wider request would have every beam read beam 0's pages.

        A batch borrowing a page table requires ``row``: its rows are indexed by whatever
        slot the lender gave the request, and one chosen here would address a different
        request's entries.
        """

    def remove_kv_cache(self, kv_cache: _KVCache) -> None:
        """Free the cache's row, leaving a hole for reuse."""

    @property
    def num_index_tables(self) -> int: ...
    def index_table(self, table_id: int) -> list[IndexTableRequest]:
        """Every request this table satisfies.

        A request's buffer order is its own mapping into the buffer dimension, so the first
        buffer of each sits at entry zero. Requests whose converted indices coincide are
        merged onto one table, which is how a quantised layer's data and block scales come
        to share one.
        """

    def index_table_slot(self, layer_id: LayerId, role: DataRole) -> tuple[int, int]:
        """``(table, entry)`` serving this buffer.

        The normal query: a kernel asks where its indices are and at what entry, then binds
        :meth:`index_table_base`. Iterating :meth:`index_table` is for introspection.
        """

    def index_table_base(self, layer_id: LayerId, role: DataRole) -> MemAddress:
        """What a kernel adds to a converted index for this buffer.

        Every buffer of a request shares it, because the request's own offsets are already
        in the entries.
        """

    def index_mode(self, layer_group_id: LayerGroupId) -> PageIndexMode:
        """Page index mode of one life cycle, as chosen at construction.

        Left unset at construction, each life cycle takes the smaller table: ``SHARED``
        where the manager reports it always available, ``PER_LAYER`` otherwise. Fixed here
        because a converted table's shape is baked into a captured graph, so it cannot
        follow a member's scratch state.
        """

    def row_of(self, kv_cache: _KVCache) -> int:
        """Row of a member, or ``-1``."""

    @property
    def max_batch_size(self) -> int: ...
    @property
    def max_blocks_per_seq(self) -> int: ...
    def resize(
        self,
        capacities: Sequence[int | None],
        history_lengths: Sequence[int | None],
        stream: CudaStream,
    ) -> list[bool]:
        """Resize every member, then publish once.

        One result per row, in row order: a member rolls back on its own, so a batch can
        partially fail. A table published after a partial failure describes the
        post-rollback state.
        """

    def publish(self, stream: CudaStream) -> None:
        """Copy the staged rows to the device, enqueuing on ``stream``.

        Page indices are published as stored -- level-relative, unscaled, and with
        ``BAD_PAGE_INDEX`` preserved rather than folded to a safe slot. A sparse table
        numbers pure-history and input-token entries in different levels' pools, so only
        the sentinel separates an absent block from slot zero; the dense path's scaled
        block offsets cannot serve here for that reason.

        A borrowed page table is copied too, not read in place: the snapshot is the whole
        point, since the host has to be free to write the next step's rows long before the
        kernels reading this step's have retired. Borrowing costs one strided transfer per
        layer group instead of one linear transfer for all of them, because the lender's
        rows are further apart than they are wide.

        Two stages. The host-to-device fill runs on a stream of the batch's own, so it
        overlaps the previous step -- the host enqueues this while that step's forward is
        still executing. ``stream`` then carries one linear device-to-device copy into the
        tables the fetch reads. Everything expensive is in the first stage, and only the
        second waits for anything.

        Raises if the previous publish has not been waited on. This call stages the
        watermarks and views itself, so it would be writing into the same hazard it is
        meant to close. It refuses rather than waiting because a wait here covers only its
        own write: a member mutated before the call has already raced by then, and waiting
        would hide that. :meth:`resize` publishes, so republishing after one needs
        :meth:`wait_published` in between.

        **The second stage must be ordered after the kernels that read the previous
        publish.** Those live in a captured graph, so the tables cannot be double-buffered
        out of the hazard -- the capture baked their addresses, which is why the staging
        sits upstream of them instead. Issuing this on the stream the graph replays on is
        what provides that ordering today. A torn publish is not a stale answer: the cache views
        carry pointers, so it can pair one request's residency with another's lease.
        """

    def wait_published(self) -> None:
        """Block until the last :meth:`publish` has finished reading the staging rows.

        The copy is asynchronous and reads pinned memory, so it is still in flight when
        :meth:`publish` returns. Every later host write to a member's page indices -- a
        resize, a commit, a beam fork, an SWA recycle -- must be separated from it by this
        call, or that write races the copy and publishes a torn table.

        :meth:`resize` already does it, which is why driving a batched step through that
        rather than through the members is the sanctioned path. For the operations with no
        ``Batch`` equivalent the simple rule is to call this once at the top of a step: it
        is idempotent and free once the copy has landed.
        """

    def page_table(self, layer_group_id: LayerGroupId) -> DeviceArray:
        """``int32[max_batch, max_blocks_per_seq]`` base page indices for one group.

        A DLPack array over memory the ``Batch`` owns; it keeps the ``Batch`` alive for as
        long as the array exists. Always packed, whatever the host rows it was published
        from look like.

        An unoccupied row reads as ``BAD_PAGE_INDEX`` throughout until the first publish.
        With a borrowed table it then holds whatever the lender had in that row, since the
        transfer carries every row rather than paying for one per member.
        """

    def history_lengths(self) -> DeviceArray:
        """``int32[max_batch]`` per-row history length, in tokens.

        Published by the same transfer as the page tables, because it bounds which entries
        of a row are valid: pairing one step's table with another step's lengths would read
        past the end of a row that may since have been reused.

        The counts a consumer wants are all derived from this and :meth:`capacities`:
        ``num_blocks`` is ``history_length // tokens_per_block``, while the sparse resolve
        works in units and divides by ``unit_tokens`` instead. Publishing the source rather
        than the three derived counts keeps one truth.

        A free row reads zero -- a request with nothing eligible.
        """

    def capacities(self) -> DeviceArray:
        """``int32[max_batch]`` per-row capacity, in tokens. See :meth:`history_lengths`."""

class KVCacheManager:
    def __init__(
        self,
        config: KVCacheManagerConfig,
        event_manager: KVCacheEventManager | StreamingEventSink | None = None,
        cold_page_codec: IKvCacheColdPageCodec | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    def clear_reusable_blocks(self) -> None: ...
    def get_mem_pool_base_address(
        self, layer_id: LayerId, data_role: DataRole, index_mode: PageIndexMode | None = None
    ) -> MemAddress: ...
    def get_sparse_cache_max_units(self, layer_id: LayerId) -> int:
        """Cells a request's cache holds per buffer of this layer, in selection units.

        Counts both populations -- the history it caches and everything in flight -- so the
        lease follows from this alone. Only one unit is ever in flight, so the surplus above
        the selection width is retention rather than headroom. Zero when the layer has no
        sparse fetch path.
        """

    def get_sparse_cache_units_per_page(self, layer_id: LayerId) -> int:
        """Selection units one page of this layer's buffers holds; one when a unit is a page.

        Distinct from :attr:`PageIndexConverter.expansion`, which is how much a buffer's page
        *index space* is stretched by ``tokens_per_block_override``. A sparse buffer does not
        stretch its index space -- that stays 1 -- while this counts the sparse cache's own
        granularity, set by the layer's ``sparse_cache.unit_tokens``.
        """

    def get_sparse_host_bytes(self) -> int:
        """Host bytes held on behalf of sparse buffers, across every pool group.

        The hot level alone: sparsity is a property of a GPU-tier pool. These bytes are a
        different medium from the one :meth:`get_quota` measures, so they are drawn from no
        quota and appear in no other total a caller can read -- report them separately when
        accounting for what a deployment pins.

        A sparse pool's row count follows from the device-backed pools of its group, since
        a slot spans every pool, so this moves only when the hot level is resized.
        """

    def get_sparse_cache_buffer_count(self, layer_id: LayerId) -> int:
        """Sparse buffers of this layer's life cycle, one per ``(layer, role)``.

        Each has its own cells, so a lease is this many runs of
        :meth:`get_sparse_cache_max_units` cells.
        """

    def get_sparse_cache_base_address(self, layer_id: LayerId, data_role: DataRole) -> MemAddress:
        """Base of the sparse cache arena backing this buffer. Raises if it has no fetch path.

        The indices from :meth:`fetch_sparse_pages` and :meth:`prepare_sparse_write_indices`
        are relative to this base. The arena holds the GPU-resident units that model kernels
        read and write.

        Distinct from :meth:`get_mem_pool_base_address`, which names this buffer's *pages*
        for every buffer alike -- what the transceiver and the KV transfer path address.
        That meaning does not change when a buffer is sparse; the arena is a second
        allocation, so it gets a second accessor rather than overloading the first.

        Role-dependent: the arena mirrors the life cycle's hot pool group, so buffers of
        different sizes -- data and its block scales, say -- sit in regions of their own, and
        an index is relative to the region of the buffer it was fetched for. Buffers sharing
        a pool share a region, and the index already carries which of them it belongs to.
        """
    def is_sparse(self, layer_id: LayerId, data_role: DataRole) -> bool:
        """Whether this buffer was declared ``BufferConfig.is_sparse``.

        Every buffer locks its pages at the hot level, sparse or not, so
        :meth:`_KVCache.get_base_page_indices` reports one level's numbering throughout.
        What a sparse buffer changes is the *medium*: its pages are host memory,
        device-mapped, in a pool of their own beside the layer's device-backed ones.

        That level is not what a caller addresses to read a unit resolved by
        :meth:`fetch_sparse_pages`. Those live in the request's lease on the life cycle's
        cache arena, which is a separate allocation reached through
        :meth:`get_sparse_cache_base_address`.
        """

    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def prepare_sparse_write_indices(
        self,
        batch: Batch,
        buffers: Sequence[BufferId],
        out: DeviceArray,  # int32[batch, len(buffers)]
        stream: CudaStream | None = None,
    ) -> None:
        """Give each request's in-flight unit a cell, and report where to write it.

        The unit between the two watermarks is the one the model is about to produce. It
        has no host copy yet -- a unit reaches its page only when :meth:`_KVCache.resize`
        carries the watermark past it -- so nothing is fetched here. The call assigns a
        cell and leaves its bytes to the model; reading one before the model writes it
        yields whatever the cell last held.

        Separate from :meth:`fetch_sparse_pages` because the orders differ. A fetch needs
        the selection, which the indexer produces from the previous layer's output; a write
        needs only the watermarks. So this can run first, which is the order the model
        requires, and it needs no top-k.

        Idempotent while the watermarks and the reservation hold. The resolve probes
        residency before taking a victim, so a later call -- including the one inside
        :meth:`fetch_sparse_pages` -- finds the entry and reuses it rather than assigning a
        second. The entry is pinned for as long as its unit is in flight, so no fetch can
        evict it, and a fetch never copies over it: a unit at or above the watermark has no
        written-back host page, so there is nothing it could be refilled from.

        Args:
            batch: whose watermarks and cache views are published. A row's cache must hold
                the reservation taken at :meth:`_KVCache.resume`.
            buffers: exactly one layer's sparse buffers, in the cache's order -- the same
                rule and the same order as :meth:`fetch_sparse_pages`.
            out: int32 ``[batch.max_batch_size, len(buffers)]``, written in full.
                ``out[b][i]`` is the index of the in-flight unit's cell for ``buffers[i]``,
                and ``BAD_PAGE_INDEX`` where the row holds no reservation or has nothing in
                flight. One entry per request, not per query token: :meth:`_KVCache.resize`
                bounds ``capacity - history_length`` to a single token, so at most one unit
                is ever in flight.

                ``BAD_PAGE_INDEX`` is an ordinary outcome, not an error. A unit is in flight
                only once it is whole: with ``tokens_per_block_override`` of 4, that is one
                step in four, which is also the only step on which a ratio-4 compressor has
                produced anything to write. With one token per unit it is every step.
            stream: the CUDA stream to enqueue on.

        Indices are in the same expanded index space :meth:`fetch_sparse_pages` produces
        and relative to the same base -- :meth:`get_sparse_cache_base_address` for that
        buffer, **not** :meth:`get_mem_pool_base_address`. The latter names the buffer's
        host-backed *pages*; the cells live in a separate GPU arena.

        Call it after publication and before the model's KV writes, ordering the two
        through the stream. Regenerate the output whenever a watermark or a reservation
        changes, and keep it unmodified until its consumers are done. Complete the writes
        before advancing history through :meth:`_KVCache.resize`, which is what writes a
        finished unit back to its host page.

        Device-driven and stream-ordered, with static shapes, so a CUDA graph can capture
        it.


        Order it after batch publication and before consumers, on the same stream or
        through events across streams. The call reads no device values on the host, and
        with fixed batch and output shapes it is CUDA-graph-capturable.

        The entry is shared by all of the layer's sparse buffers, but each buffer's index
        is relative to its own :meth:`get_sparse_cache_base_address`, including when data
        and scales sit in different pools. The indices carry request and buffer placement;
        they are neither request-local cache-entry numbers nor byte addresses.
        """

    def fetch_sparse_pages(
        self,
        batch: Batch,
        buffers: Sequence[BufferId],
        selected: DeviceArray,  # int32[batch, topk]
        out: DeviceArray,  # int32[batch, len(buffers), topk]
        stream: CudaStream,
    ) -> None:
        """Resolve a batch's selections against each buffer's per-request history cache.

        Device-driven and stream-ordered throughout: the selection is read from device
        memory, so the host neither inspects it nor decides anything, and the call
        enqueues onto ``stream`` and returns. Order it against the selection kernel by
        sharing a stream, or across streams with an event. Every shape is static across
        steps, so the call is capturable in a CUDA graph.

        ``buffers`` must be exactly one layer's sparse buffers, in the order
        :meth:`get_mem_pool_base_address` numbers them. One layer, not one layer group:
        a layer's selection is computed from the previous layer's output, so a step's
        selections are not all known at its start. Consecutive layers that share a
        selection still get a call each, so the first layer's copies start as early as
        possible; pass the same ``selected`` to each.

        The whole layer, not a subset, because the resolve marks entries resident on
        behalf of every buffer at once: filling only some would leave the rest stale while
        the metadata claimed otherwise.

        ``selected`` holds block ordinals, ``topk`` uniform across requests and ``-1``
        past a request's valid count. The page table and the per-request count of
        eligible blocks come from ``batch``, whose row order ``selected`` and ``out``
        both inherit.

        Eligible means below the history watermark: a block at or above it holds input
        tokens whose host page has not been written back yet, so selecting it would read
        stale bytes. Not a memory-safety problem -- the page table is uniform, so the
        ordinal resolves to a real page of the same request -- but a wrong answer. The call
        cannot detect an ineligible ordinal, so naming only eligible ones is the selector's
        responsibility.

        Each selected unit is looked up in the owning request's cache for that layer.
        A hit is reused in place; a miss is copied from the locked host page into a slot
        taken from the free list, evicting least-recently-used when it is empty.
        Reservation at :meth:`_KVCache.resume` guarantees room for a full selection
        width, so a fetch always satisfies its own layer, and within a step a fetch may
        evict entries of layers already processed but never one a later kernel still
        needs.

        Copies leave the host page authoritative rather than moving it, so evicting a
        fetched cell moves no data. The copies are a snapshot, which is sound only because a
        block below the watermark is already written back and immutable. A cell holding an
        in-flight block is the one exception, and is never an eviction candidate.

        The in-flight unit is reserved internally but is not appended to ``out``. Use
        :meth:`prepare_sparse_write_indices` before the model's KV writes to obtain its
        per-buffer physical write indices independently of the selection.

        ``out`` is written in full and is **compact**: entry ``[b][i][k]`` is the cache
        index holding ``selected[b][k]`` for ``buffers[i]``, and ``BAD_PAGE_INDEX`` where
        that slot carries no selection. Padding stays where the
        selection put it rather than being packed out, so slot ``k`` of ``out`` and slot
        ``k`` of ``selected`` describe the same unit.

        Compact, in the selection's own order, rather than a table indexed by unit
        ordinal, because that is the shape sparse attention kernels take: a token-sparse
        kernel is handed the index list in place of its page table and reads one unit per
        slot, and a block-sparse kernel takes the list beside a page table it still uses.
        None indexes a table by ordinal, so producing one would only oblige the caller to
        compact it again.

        The buffer dimension is a dimension rather than a separate array per buffer
        because attention kernels commonly want one tensor covering K and V. It cannot be
        dropped: a layer's buffers share the resolve, but not their cells. A lease gives
        each buffer its own contiguous run of cells, so one selection resolves to a
        different index per buffer, and each needs its own row.

        Indices are in the expanded index space :class:`PageIndexConverter` produces, and
        are relative to :meth:`get_sparse_cache_base_address` for the buffer they were
        fetched for -- **not** to :meth:`get_mem_pool_base_address`, and not to a cache
        level. The pool names the buffer's pages, which for a sparse life cycle are host
        memory; the cells a kernel reads live in a separate GPU arena, with a region per
        pool. Pairing these indices with the pool base reads the wrong allocation.

        The selection is per layer, not per KV head: every head of a layer shares one
        resident set. Algorithms that let different KV heads select different units are
        therefore not expressible against this cache.

        **Decode only.** ``selected`` carries one row per batch row, so a request
        contributes a single selection per call. Sparse attention kernels index their
        selection by *query token*, which coincides with the request only while a request
        contributes one token -- that is, during generation.

        The restriction is not just the row count. The resolve makes the selected units
        resident in the request's cache, which holds ``max_cached_units``; residency is a
        property of the request, not of a query token. A prefill chunk of N query tokens
        each choosing their own units has a union of up to N times the selection width, so
        the reservation could no longer cover a call, and a unit evicted for a later query
        token would invalidate an index already returned for an earlier one. Widening this
        to one row per token would make that reachable rather than fix it.

        Serving prefill from a host-resident history is an open design question; see
        SPARSE_ATTENTION_DESIGN.md.
        """

    def get_page_index_scale(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_converter(
        self, layer_id: LayerId, data_role: DataRole
    ) -> PageIndexConverter: ...
    def create_kv_cache(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
        id: Any = None,
        custom_priority_callback: Callable[[int, Any], Priority] = ...,
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
    ) -> _KVCache:
        """Create a suspended cache with a prefill-to-generation boundary.

        On the C++ backend, ``expected_prompt_length`` also determines which full
        prompt blocks are shared by beams. Pass the actual full prompt length for
        beam search if ``input_tokens`` is absent or shortened for reuse matching.
        It defaults to the non-empty input length; without either value, the beam
        sharing boundary is zero and the statistics boundary is unset. When set,
        it also marks generation-phase allocation stats.
        """
        ...
    def probe_reuse(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> int: ...
    def probe_first_new_block_key(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> bytes | None: ...
    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool: ...
    def get_quota(self, cache_level: CacheLevel) -> int: ...
    def get_storage_statistics(self, cache_level: CacheLevel = ...) -> list[StorageStatistics]: ...
    def get_life_cycle_pool_group_indices(
        self, cache_level: CacheLevel = ...
    ) -> list[PoolGroupIndex]: ...
    def get_committed_stats(self) -> KVCacheStatsDelta: ...
    def get_and_reset_iteration_stats(self) -> dict[LifeCycleId, KVCacheIterationStatsDelta]: ...
    def get_and_reset_ssm_snapshot_iteration_stats(
        self,
    ) -> dict[LifeCycleId, SsmSnapshotIterationStatsDelta]: ...
    def record_request_suspended(self) -> None: ...
    def record_request_resumed(self) -> None: ...
    def get_and_reset_iteration_suspend_resume_stats(self) -> tuple[int, int]: ...
    def get_and_reset_iteration_disk_prefetch_blocks(self) -> int: ...
    def get_and_reset_iteration_cached_tokens_by_level(self) -> list[int]: ...
    def get_and_reset_iteration_reused_blocks_by_level(
        self,
    ) -> dict[LifeCycleId, ReusedBlocksByLevel]: ...
    def get_and_reset_iteration_peak_block_stats(
        self, cache_level: CacheLevel
    ) -> Sequence[PoolGroupPeakBlockStats]: ...
    def get_and_reset_iteration_peak_block_stats_by_level(
        self,
    ) -> Sequence[Sequence[PoolGroupPeakBlockStats]]: ...
    def mark_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def get_dirty_stats_kv_cache_ids(self) -> set[int]: ...
    def mark_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def is_stats_excluded(self, kv_cache_id: int | None) -> bool: ...
    @property
    def cache_tier_list(self) -> Sequence[CacheTier]: ...
    @property
    def tokens_per_block(self) -> int: ...
    @property
    def event_manager(self) -> Any | None: ...
    @property
    def init_config(self) -> KVCacheManagerConfig: ...
    @property
    def allow_seq_rebasing(self) -> bool: ...
    @property
    def enable_partial_match(self) -> bool: ...
    @property
    def enable_partial_commit(self) -> bool: ...
    def supports_index_mode(
        self, mode: PageIndexMode, layer_group_id: LayerGroupId | None = None
    ) -> bool | None:
        """``None`` means it depends on the request.

        Given a layer group the answer is definitive for every life cycle that cannot hold
        scratch slots -- anything without a sliding window -- which the collapsing form
        cannot express. Configuration only, so it never varies with what a request is doing.
        """
    @property
    def num_layers(self) -> int: ...
    @property
    def layer_ids(self) -> Iterator[LayerId]: ...
    def get_layer_group_id(self, layer_id: LayerId) -> LayerGroupId: ...
    @property
    def layer_grouping(self) -> Sequence[Sequence[LayerId]]: ...
    @property
    def all_buffer_ids(self) -> Iterator[BufferId]: ...
    def get_aggregated_pages(self, buffers: Iterable[BufferId]) -> Iterator[AggregatedPageDesc]: ...
    @property
    def pool_group_descs(self) -> Sequence[PoolGroupDesc]: ...
    def clamp_max_seq_len_for_mem(self, batch_size: int, token_num_upper_bound: int) -> int: ...
    def adjust(self) -> None: ...
    @property
    def need_adjustment(self) -> bool: ...
    @property
    def commit_min_snapshot(self) -> bool: ...

def exact_div(x: int, y: int) -> int: ...
def typed_range(*args: int) -> range: ...
