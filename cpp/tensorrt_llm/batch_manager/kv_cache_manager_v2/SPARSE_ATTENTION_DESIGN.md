# Sparse attention in KVCacheManagerV2

Status: design. No implementation yet. Part of the Python surface is declared in
`tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi` and is behind this document.

Read `AGENTS.md` in this directory first for the storage, locking and concurrency model
this builds on.

## Problem

Sparse-attention decoding reads only a small, per-step, per-layer subset of a sequence's
history. Keeping all of that history in GPU memory wastes the resource the sparse
algorithm exists to conserve: a request's memory bill still grows with its full context,
so decoding hits a capacity wall long before it runs out of compute.

The goal is to hold history on the host tier and let GPU residency be bounded by the
selection width rather than by context length.

Two existing properties make this cheap:

- The host tier is registered `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP`
  (`utils/hostMem.cpp:305`), so a kernel can read a host page directly.
  `isGpuAccessibleMemory` (`storageManager.cpp:138`) already relies on this for the
  cold-page codec, which encodes and decodes against host memory with no staging buffer.
- Selection output is a device array. If everything the fetch path needs is also on the
  device, the host never enters the loop, and the operation is stream-ordered and
  capturable in a CUDA graph.

## Vocabulary

| term | meaning |
| --- | --- |
| page | the KVCM2 allocation, locking and migration unit: `tokens_per_block` tokens |
| buffer | one layer's slice of a page; several coalesce into one page |
| selection unit | what the backend selects; a factor or a multiple of `tokens_per_block` |
| expansion | selection units per page, `tokens_per_block / tokens_per_block_override` |
| pure history | a block whose ordinal is below the `set_history_length()` watermark |

## Scope

Sparse-attention families differ in what they select, and the difference is load-bearing:

| family | selection unit | models |
| --- | --- | --- |
| MSA | block, 16 × 128 tokens (`minimax_m3/common.py:70`) | MiniMax-M3 |
| NSA, MoBA | block | DeepSeek-V4-Flash, Kimi |
| Quest | page | training-free, any model |
| DSA | **token**, k = 2048 | DeepSeek-V3.2, V4.1, GLM-5.x |

Block-level families work with the selection unit equal to one page and need nothing
special. DSA needs a sub-page unit, which this design supports through the selection-unit
mechanism below, at the cost of a finer cache entry. First implementation should target
the block-level families; DSA follows without an API change.

DeepSeek-V4's compressed pools are worth noting as a middle case: per-layer
`compress_ratios` with `compressed_block_sizes = tokens_per_block // ratio`
(`deepseek_v4/cache_manager.py:370`), so at ratio 128 a compressed page holds exactly one
entry and the selection unit is already a page.

## Shape of the design

1. **Storage tiers are unchanged.** No new tier or medium. Level 0 is GPU; level 1 must be
   `HOST_MEM` for a deployment using sparse buffers.
2. **`BufferConfig.is_sparse` is a per-buffer flag** and changes where a page locks.
   A sparse buffer requires a lifecycle of its own. Sparse buffers of the same size still
   coalesce with each other.
3. **Every page of a sparse buffer locks at level 1**, regardless of what it holds. Lock
   location is a property of the buffer, not of the block.
4. **All GPU residency is the per-request cache.** A sparse buffer has no separately
   managed GPU pages: both the block being written and the history a step selects occupy
   cache entries, drawn from the same reservation and replaced by the same LRU.
5. **The host page is authoritative once the block is finalized.** A block is flushed to
   its host page at `commit()`, after which its cache entry is clean and evictable. Before
   that the entry is dirty and ineligible for eviction.
6. **Extraction is pluggable**, because KVCM2 buffers are opaque bytes and a sub-page unit
   cannot be located without layout knowledge.

## Locking and flush

`acquirePageIndex` already writes the page's *current* slot id with no level assumption
(`page.cpp:401`):

```cpp
int old = kvc->updateBasePageIndex(
    mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, slotIdToPageIndexValue(pg.slotId()));
```

so `getBasePageIndices()` reports level-1 slot ids with no change, and because every page
of a sparse buffer locks there, the table is **uniform**: every entry is a level-1 slot id.
Nothing resolves against the GPU pool, and no caller has to know which rule applies to
which ordinal.

A block's data reaches its host page by a **flush**: a copy on the owning `KvCache`'s
stream that does **not** evict the GPU entry, which stays cached and merely becomes clean.

The governing rule is that **a block must be flushed before it can be selected**. Selectable
means below the watermark, so `setHistoryLength()` is the deadline and must flush anything
still dirty that the watermark has passed.

`commit()` flushes earlier, and should: only finalized pages enter the radix tree, so the
host page is the artifact a later request reuses. But commit is an optimisation, not the
guarantee — `stopCommitting()`, a `CommitState` other than `ALLOWED`, or a request with
reuse disabled all mean a block may never be committed. Relying on commit alone would leave
such a block dirty indefinitely: either pinned until the request ends, exhausting the cache,
or evicted while its host page is still empty.

`history_length` therefore keeps two jobs: it defines selection eligibility, and it is the
flush deadline.

### Dirty entries

A block being written is dirty until its flush completes. Three facts keep this from being
a general write-back cache:

- **At most one dirty entry per `(request, layer)`** — the block currently being filled.
- **It is private.** Only finalized pages enter the radix tree, so an unflushed block has
  exactly one owner: no aliasing, no cross-request coherence, no concurrent writers.
- **Its lifetime is bounded** by one block fill, and it is the most-recently-used entry
  throughout, since it is written every step.

Eviction eligibility is therefore the single predicate *clean*, and the bit flips once per
block lifetime. There is no evict-a-dirty-entry path to implement.

## Per-request reservation

Two separate reservations, easily confused.

**Host tier.** Every block of the sequence has a locked host page, so the count is
`divUp(capacity, tokens_per_block)` — the existing per-sequence locking logic, unchanged.
Only the level differs. This is the payoff of uniform locking: nothing new to size, and no
notion of "history pages" distinct from the rest.

**GPU tier.** The cache, bounded by `max_cached_units` (**N** in *Sizing*, two to four times
the selection width), holding

```
min(N, history_length / tokens_per_block)
```

entries, so a short sequence caches little and a long one saturates. One further entry is
reserved for the block currently being written, which is dirty until flushed and so cannot
be an eviction candidate; keeping it outside N preserves the history cache's capacity
guarantee.

Reservation happens at `resume()` and fails there if capacity is unavailable; the fetch
path never has to handle exhaustion.

### Why per-request pages give per-layer caches for free

A page already carries one buffer per coalesced layer. Reserving N pages therefore gives
**each layer N independent cache entries**, with no new allocation granularity and no
buffer-level free list. Layer L can only ever occupy position L within a page, so the
per-layer free list is just "which of my N pages is free for layer L".

The consequence is that a cache entry is `(page, layer)`, not `page`. One reserved page
may hold different layers' data for entirely different ordinals. This breaks an assumption
that runs through KVCM2 — that a page is both the unit of allocation *and* the unit of
identity — and code reasoning about "the block this page holds" has no answer for a sparse
cache page.

When the selection unit is smaller than a page, a page holds `expansion` units per layer,
so N pages give `N * expansion` unit-slots per layer and the entry is `(page, layer, unit)`.

## Device-side cache

Per `(request, layer)`, managed entirely by kernels: a resident set, an LRU queue and a
free list.

**Scope is per request.** Two requests sharing a prefix do not share a cached copy. This
matches HiSparse and closes a soundness hole by construction: entries are reachable only
through their own request's page table, so a host slot can never be recycled underneath a
surviving entry. A global, provenance-keyed cache would give cross-request sharing for
free via the radix tree, but needs generation counters or an invalidation pass; deferred.

**Replacement is true LRU**, not "evict whatever was not selected this step". Entries that
hit are promoted above newly fetched misses, so a unit selected repeatedly outranks a
first-time selection and one-off selections leave first. Retention across steps is the
point: HiSparse reports a large miss-rate reduction from holding hot entries beyond the
current top-k, which is only possible when the cache is larger than the selection width.

**Residency is tracked by a reverse map** — a dense array indexed by slot `0..N-1` holding
the resident unit ordinal, N entries per `(request, layer)` — not a forward map from
ordinal to slot. The two scale with different things:

| | size | at 1 M context, 256 requests, 64 layers |
| --- | --- | --- |
| forward, ordinal → slot | batch × layers × **max_ordinals** | hundreds of MB, and unbounded in context |
| reverse, slot → ordinal | batch × layers × **N** | proportional to cache already bought |

The forward map is proportional to context length, the quantity this feature exists to
stop paying for. The dense reverse form is also what the LRU order and free list want.

### Probe index

Answering "is this ordinal resident, and where" against a dense reverse map would be O(N)
per selection. The fetch kernel therefore builds a **hash over the resident set**, keyed by
ordinal and valued by slot, in shared memory at the top of each launch, and discards it at
the end. Two properties follow from it being ephemeral: there are no deletions, so no
tombstones and none of the maintenance that made a persistent device hash map unattractive;
and nothing has to stay consistent across steps or graph replays.

Hashing the *resident* set rather than the *selected* set is deliberate. Hashing the
selected set and walking residents also costs O(N + k), but the miss list is not known until
that walk finishes. Hashing residents means one probe per selected unit decides hit or miss
immediately, so PCIe copies can be issued before any LRU bookkeeping runs — and the probe
returns the slot for a hit, so no second pass is needed to resolve it.

Open addressing with linear probing, capacity `C = next_pow2(2N)` so the load factor stays
at or below one half, which also guarantees an unsuccessful probe terminates on an empty
entry.

```cpp
constexpr int32_t kEmpty = -1;
int32_t*  keys;     // [C] unit ordinal
uint16_t* vals;     // [C] slot index, N <= 65535
uint32_t* hitMask;  // [(N+31)/32] resident was selected this step

__device__ inline uint32_t slotOf(int32_t key, uint32_t mask) {
    return (static_cast<uint32_t>(key) * 0x9E3779B1u) & mask;   // Fibonacci
}
```

Keys and values are separate arrays rather than packed 64-bit entries: a `__syncthreads()`
separates build from probe, so no thread reads an entry mid-insert and the value needs no
atomicity, keeping the CAS 32-bit. Multiplicative hashing rather than identity, because
ordinals are small and dense, and dense runs make an unsuccessful probe walk the whole run.

Build, one thread per resident slot — this pass reads the reverse map, which the kernel
must read anyway for LRU and free-slot bookkeeping, so it adds no global traffic. Ordinals
are unique within a `(request, layer)`, so threads contend only on table-position
collisions:

```cpp
for (int s = threadIdx.x; s < N; s += blockDim.x) {
    int32_t ord = reverseMap[s];
    if (ord == kBadOrdinal) continue;              // free slot
    uint32_t h = slotOf(ord, C - 1);
    while (atomicCAS(&keys[h], kEmpty, ord) != kEmpty)
        h = (h + 1) & (C - 1);
    vals[h] = static_cast<uint16_t>(s);
}
__syncthreads();
```

Probe, one thread per selected unit. `hitMask` then drives eviction: residents with a clear
bit are evictable, in LRU order.

```cpp
for (int i = threadIdx.x; i < k; i += blockDim.x) {
    int32_t ord = selected[i];
    if (ord < 0) continue;                         // padding
    uint32_t h = slotOf(ord, C - 1);
    int32_t  key;
    while ((key = keys[h]) != kEmpty && key != ord)
        h = (h + 1) & (C - 1);
    if (key == ord) {
        uint16_t s = vals[h];
        outSlot[i] = s;
        atomicOr(&hitMask[s >> 5], 1u << (s & 31));
    } else {
        outSlot[i] = kMiss;
    }
}
```

At 6 bytes per entry plus the mask, shared-memory use is ~0.8 KB at N = 64 (MSA, k = 16)
and ~6 KB at N = 512 — negligible. At N = 8192 (DSA, k = 2048) it reaches ~96 KB, which
fits only in Hopper/Blackwell dynamic shared memory and costs occupancy. Occupancy matters
less here than usual, since the kernel is bound by PCIe copy latency and one CTA per
`(request, layer)` is an acceptable shape, but at that size prefer one of: storing only the
slot per entry and comparing against `reverseMap[slot]` (2 bytes per entry, one indirection
per probe step); tiling the resident set into two passes; or a bitset indexed by ordinal,
which is affordable exactly at the granularity where N is large only because units are
small.

## Extraction interface

KVCM2 buffers are raw bytes with no layout information, so locating a selection unit inside
one requires the owner's knowledge. Two mechanisms, chosen per buffer, serving two
different situations.

### Fast path: a layout value

`SelectionUnitLayout` is fixed per buffer and consumed once at configuration, so it needs
no dispatch and is not an interface. It is a value on `BufferConfig`:

```cpp
struct SelectionUnitLayout {   // buffer-relative
    uint32_t numChunks;      // one per head; 1 for token-major
    uint32_t chunkBytes;     // contiguous bytes of one unit within one chunk
    uint32_t chunkStride;    // distance between chunks
    uint32_t unitStride;     // distance between units within a chunk
};
```

It is **buffer-relative and carries no page offset**, for an ordering reason: `BufferConfig`
is an input to the `KVCacheManager` constructor, while cold-page placement is not decided
until the codec's `configure()` runs *during* construction. A cold-page offset could not be
supplied there. It is also the only part the model integration knows — the internal shape of
its own buffer.

KVCM2 supplies the rest, which is why **the fast path requires the default codec**. That
codec's cold page is a direct concatenation of buffers, so KVCM2 knows each buffer's offset
within it; the hot-slot offset comes from `BufferAttr` as usual. No query of the codec, and
no downcast, is needed.

Because the default codec relocates buffer bytes verbatim, a buffer's cold layout equals its
hot layout, and a device cache slot's buffer has the same structure as any hot buffer —
`expansion` unit positions. The copy is therefore symmetric, one set of strides on both
sides, differing only in unit index:

```
src = coldPageBase + coldBufferOffset + c*chunkStride + srcUnit*unitStride
dst = hotSlotBase  + hotBufferOffset  + c*chunkStride + dstUnit*unitStride
```

The destination stays in natural layout rather than being repacked, so the attention kernel
sees exactly what it sees for an ordinary page.

| cold layout | numChunks | chunkBytes | chunkStride |
| --- | --- | --- | --- |
| token-major, contiguous | 1 | `unit_tokens * bytes_per_token` | — |
| HND `[heads][tokens][dim]` | `num_heads` | `unit_tokens * dim * esize` | `tokens_per_block * dim * esize` |

One region suffices because block-quantisation scales are registered as a separate
`DataRole`, hence a separate buffer with its own layout, size and cache entries. A buffer
packing data and scales together would need a region array; the symmetric form above extends
to that without redesign.

Worth asserting at configuration: `numChunks * chunkBytes == BufferConfig.size`, which is
already per-unit (`expandedSize = buf.size * exp`).

### Fallback: the gather interface

Not part of the first implementation — see *Staging* below. Sketched here because it shapes
the descriptor path's boundary.

A layout can only express data *movement* between two arithmetic-addressable layouts. A
custom codec, a variable-length or compressed cold representation, dequantisation, or
permutation all need code, and that code runs at step time on a stream — so unlike the
layout, it is genuinely an interface:

```cpp
class ISparseUnitGather {
public:
    virtual ~ISparseUnitGather();
    virtual bool gather(/* miss list, cold base, hot base, */ CUstream stream) noexcept = 0;
};
```

Registered per sparse buffer alongside the optional layout. The resolution rule is checkable
at configuration: a layout present selects the fast path; absent, a gatherer is required;
neither is a configuration error.

The two deliberately share no base class. They have almost nothing in common — the layout is
inert data, the gather is a stream-taking device call — and a shared base would turn the one
useful composition, a single implementation serving descriptors for most buffers and a gather
for one, into a diamond needing virtual inheritance. If a shared `configure()` later proves
necessary, adding a base is a small internal refactor; neither type crosses an ABI boundary.

### Kernel structure

The descriptor path is **one kernel**. KVCM2 resolves selections, updates LRU, allocates
slots and performs the copies itself, because a layout is data it can interpret rather than
code it must call. Nothing is split and nothing needs fusing.

The gather path is **two kernels**: KVCM2's kernel resolves selections, updates LRU and
allocates slots, writing a miss list to a device array; the implementation's kernel performs
extraction and copy. This is the same shape as `encode(pairs, stream)` and needs no new
machinery.

Fusing *the gather path* is feasible later, and the mechanism matters:

- A `__device__` **function pointer will not work** across the boundary. Device function
  pointers are module-scoped, so a pointer obtained from a separately compiled `.so`
  cannot be called from a kernel in another module. Only same-binary device linking
  (`-rdc=true`) would permit it, which defeats the plugin model.
- The route that does work is returning the extraction function as **LTO-IR**, compiling
  KVCM2's kernel to LTO-IR too, and linking with **nvJitLink** for real inlining. There is
  in-tree precedent for runtime compilation in
  `kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/` (`nvrtcWrapper`,
  `compileEngine`, `cubinObj`) with module loading via `common/cudaDriverWrapper`.
- Compile at configuration, cache by (layout, arch), never per step.

Fusion may never be needed. It applies only to compressed or otherwise non-addressable cold
representations, which are the minority; the design is graph-capturable, so on replay a
second launch costs well under a microsecond against a millisecond-scale step whose copies
dominate.

## Staging

The three mechanisms above are deliberately ordered so each stage is useful alone and none
blocks the next:

| stage | delivers | needed when |
| --- | --- | --- |
| 1 | descriptor path, single fused kernel, no plugin interface at all | always — the flush is a verbatim copy, so cold layout equals hot layout |
| 2 | `ISparseUnitGather`, two kernels on that path | compression, or any other non-arithmetic cold representation |
| 3 | LTO-IR + nvJitLink to re-fuse stage 2 | stage 2's second launch measurably costs something |

Stage 1 needs no extension point, so the first implementation carries no plugin machinery,
no registration, and no dispatch. `ISparseUnitGather` is introduced only alongside the first
codec that requires it, which is also the first point at which its shape can be validated
against a real consumer rather than guessed.

Compressing history on the way to host belongs to stage 2. It is attractive — history is
the bulk of the footprint and is written once and read rarely — but it breaks the identity
between cold and hot layout that the descriptor path depends on, and raises a lossy-versus-
lossless question the first stage does not need to answer.

## Fetch

One call per layer group per step, on the caller's stream:

1. read the selection: unit ordinals, uniform width, `-1` padded;
2. build the probe index over the resident set;
3. probe once per selected unit — hit with its slot, or miss;
4. allocate a slot per miss from the free list, evicting LRU when empty;
5. extract and copy each miss from the device-mapped host page;
6. promote hits above new misses in recency order;
7. write the output index list.

Steps 5 and 6 are independent. Because the probe in step 3 resolves misses directly, the
copies — the only step that touches PCIe, and so the long pole — can be issued before the
recency bookkeeping runs rather than after it.

No host involvement, so the call enqueues and returns. Every shape is static across steps.

Fetch is not the only path that allocates a cache entry: starting a new block allocates one
directly, outside any fetch, and holds it dirty until commit. Both draw from the same free
list.

### Capacity and eviction

Capacity is guaranteed by reservation, not back-pressure: a fetch can always satisfy its
own layer, because the reservation covers a full selection width. Within a step a fetch may
evict entries of layers already processed but never one a later kernel still needs, which
LRU recency delivers — layer N's entries are most-recently-used when layer N+1 fetches.

An entry is evictable only when **clean**. That predicate is what protects the block being
written, so it is a correctness bit rather than a policy hint — but it is also the only one,
and it is set once per block at commit.

### Preconditions the caller owns

Selections must name only pure-history units; `num_blocks` carries each request's eligible
count.

Naming an ineligible one is no longer a memory-safety problem. The table is uniform, so the
ordinal resolves to a real host page of the same request — but an unflushed block's host
page has no valid contents yet, so the result is stale data rather than unrelated memory.
A debug check against the flushed state can catch it; silence would be a wrong answer, not
corruption.

## One pool, one index space

The sparse pool's GPU tier holds nothing but cache entries — selected history and the block
being written alike. That is what lets the fetch emit a **single** index list covering both,
and therefore why existing attention kernels need no change: a kernel still indexes one page
table by ordinal against one pool base.

Indices are unit-granular. With `E` units per page and `N_b` buffers per page, layer L's
unit u of page p is at `(p * N_b + L) * E + u` — the expanded index space
`PageIndexConverter` already produces.

Note that `tokens_per_block_override` is only an index-conversion mechanism: it yields
`expansion = tokens_per_block / tokens_per_block_override` and scales `BufferConfig.size`
accordingly (`storage/config.cpp:174-178`, `storage/config.h:169`). Allocation, locking and
migration remain on the global `tokens_per_block`. It defines the selection unit's size and
index space, not its physical granularity.

## Batch

Every device array here is indexed by batch row, and every call must agree on which row a
request occupies. `Batch` owns that: membership, row order, the page table, `num_blocks`,
and the staging mirror. It is global, not per layer group, and carries one page table per
layer group.

**Dirty rows.** A `_KVCache` knows its `Batch` and row; any lock-changing operation marks
that row dirty, and `Batch` republishes only dirty rows. This follows the existing
`mark_stats_dirty` / `clear_stats_dirty` / `get_dirty_stats_kv_cache_ids` pattern.
Rejecting direct mutation instead was considered and rejected: `commit` changes locks and
runs every decode step, so strictness would require a batched twin for every mutator before
anything could ship. Dirty tracking degrades into a slower step rather than a wrong answer.
A debug-mode check can still flag direct mutation while batched.

**Batched resize** resizes every member and publishes once at the end. It may initially
forward to `_KVCache.resize()`; the committed contract is row order and
publish-on-completion, which survives the later coalescing work. The result is per-request,
not a single flag, since `_KVCache.resize()` already rolls back per cache; a table published
after a partial failure must reflect the post-rollback state.

**Rows are slots.** Removing a request leaves a hole to be reused rather than shifting later
rows, which would invalidate the whole table and destabilise the shapes graph capture needs.
The `_torch` layer already works this way with `index_mapper_capacity` and a `copy_idx`
gather. Membership is exclusive, and closing a member or destroying a batch with live
members needs defined behaviour, since `SharedPtr` refcounts here are non-atomic.

## Sizing

Per sparse buffer: **`num_selected_blocks`** (the selection width `k`) and **N** (the
reservation upper bound, in the same units).

N should be `2k`–`4k`, not `k`. `k` is the correctness floor — it guarantees a fetch can
satisfy its layer — but it leaves no room for retention, and retention across steps is
where the hit rate comes from. HiSparse finds `B ∈ [2k, 4k]` the useful range.

The split between cache reservation and ordinary full pages follows from `k` rather than
being configured separately. This is a decode-time feature, so a sequence holds one or two
blocks of non-history tokens at a time:

```
cache reservation  = max_batch_size * N            (N in [2k, 4k])
full pages        ~= max_batch_size * (1 or 2)
```

with `max_batch_size` from `constraints`. An explicit ratio between the regions stays
available as an escape hatch. Extending sparse attention to prefill would remove the small
constant on the non-history side and invalidate this derivation.

## Python surface

Declared or planned in `tensorrt_llm/runtime/kv_cache_manager_v2/__init__.pyi`:

- `BufferConfig.is_sparse: bool`, `BufferConfig.num_selected_blocks: int`, the reservation
  bound, and `BufferConfig.sparse_unit_layout: SelectionUnitLayout | None` —
  construction-time, per buffer.
- `KVCacheManager.is_sparse(layer_id, data_role) -> bool`.
- `Batch` — membership, row order, device arrays, `resize()`, dirty-row publication.
- `KVCacheManager.fetch_sparse_pages(batch, buffers, selected, out, stream)`.

Publishing the page table is `Batch`'s job. It must not reuse the dense path's block
offsets: those are scaled and fold `BAD_PAGE_INDEX` to slot zero, whereas a sparse table
numbers history and input-token entries in different levels' pools and needs the sentinel
to distinguish an absent block from slot zero.

Device arrays are typed by a `DeviceArray` protocol over DLPack, so callers pass torch
tensors without this package depending on torch.

## Implementation notes

Slot release is already level-aware: `Page::~Page()` releases with the page's own
`cacheLevel`, and the `kHotLevel` in `ScratchSlotLock::unlock()` is correct, since SWA
prefill scratch is GPU-only by definition.

- **`_lockHeldBlocks`**, the OOM rollback for `resize`, calls `batchedLockToGpu`. A sparse
  resize failing partway would re-lock pages at level 0, leaving the page table holding
  level-0 indices in a table everything else assumes is level-1. Failure path only, so it
  survives happy-path testing. *(Addressed: `StaleBackup` now records the page's lock level
  and `batchedLockPages` restores it.)*
- **`_unlockStaleBlocks` is not the flush hook.** It exists for SWA blocks outside the
  window, whose content is *dead*; sparse history is live and will be read. Same trigger and
  range arithmetic, different disposition — routing sparse layers through it drops the data
  instead of flushing it. It also gates on `windowSize.has_value()`, so for a full-attention
  sparse layer it fires on nothing.
- **`_shortcutSetHistoryLength`** bails out when the SWA stale range is unchanged. The flush
  range advances on a different schedule, so the comparison must include it or the fast path
  silently skips a flush — leaving a block dirty past the point where selection may name it.
- **Commit is not a substitute for the watermark deadline.** Flushing at `commit()` is worth
  doing, but `stopCommitting()`, a `CommitState` other than `ALLOWED`, and reuse-disabled
  requests all produce blocks that are never committed. The watermark hook is what makes the
  flush guaranteed.

Also: the host pool is far larger than the GPU pool, so a page index must be widened to 64
bits before multiplying by the page stride. MSA already does this
(`triton_sparse_decode.py`); QSA and XQA should be audited.

## Settled by construction

Recorded because each looks like an open question and is not:

- **`num_blocks` lives on the device, in `Batch`.** Deriving it per call from the watermark
  would produce a host value the kernel cannot use, costing an H2D per layer.
- **Eviction never copies.** An entry is evictable only once flushed, so evicting is a
  free-list push. Flushes are triggered by commit or by the watermark, never by eviction
  pressure.
- **The dirty bit is not a general pinning scheme.** One entry per `(request, layer)`,
  private because unflushed blocks are not in the radix tree, and MRU throughout its life.
  Ordering between layers still comes from LRU recency alone.
- **No stale-key invalidation.** Per-request cache scope makes it impossible.
- **`tokens_per_block_override` is not a granularity lever.** See *One pool, one index
  space*.

## Open questions

- Whether the decode-only assumption should be enforced or merely documented, given the
  sizing derivation depends on it.
- For a selection unit spanning multiple pages: whether the device slots must be
  contiguous, and how eviction handles k-at-a-time.
- Whether `N` is uniform across requests. Uniform keeps reservation a counter and shapes
  static; varying it by context length would improve hit rates for long requests at the
  cost of ragged device arrays.
- Whether rewinding past a commit is already rejected. The watermark-driven design needed an
  explicit rule against rewinding below a demoted block; with commit as the flush point that
  rule should be subsumed by existing commit semantics, but `stopCommitting()` and
  `maxRewindLen` should be checked rather than assumed.

## References

- HiSparse: [arXiv:2608.07009](https://arxiv.org/abs/2608.07009),
  [SGLang docs](https://docs.sglang.io/docs/advanced_features/hisparse_guide),
  [LMSYS blog](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- MiniMax Sparse Attention: [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
- DeepSeek-V4: [arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)
