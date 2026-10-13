# Sparse attention in KVCacheManagerV2

Target: the **decode instance of a disaggregated deployment**.

This document describes one design. Where the code has not caught up, the text says so
inline and *Migration state* below tracks it; everything else is in place.

**Implemented**: the resolve, copy, index and flush-list kernels; `Batch`; the per-request
reservation and device state; the write-back at `resize()`; `KVCacheManager.fetch_sparse_pages`.

**Pending** (see *Migration state*): sparse history currently resides at a dedicated cold
level rather than on a host-backed hot pool group, and the GPU cache is reserved from
storage slots rather than its own arena. The four steps that close the gap are listed there.

Out of scope, by decision rather than by omission: prefill, which a separate instance serves,
and speculative decoding — see *Supported scope*. Kernel fusion remains future work; so do
multi-beam requests and life cycles whose sparse buffers span more than one pool.
`ISparseUnitGather` is retired — see *Extraction interface*.

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
| selection unit | what the backend selects; a factor of `tokens_per_block`, never larger |
| in-flight | blocks above the watermark: written but not yet flushed |
| expansion | selection units per page, `tokens_per_block / tokens_per_block_override` |
| pure history | a block whose ordinal is below the history watermark set by `resize()` |

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

1. **No new tier or medium.** What varies is the *memory kind of a pool group*: a sparse
   life cycle's level-0 pool group is backed by host memory, a dense one by GPU memory.
   Level 1 stays an ordinary codec target for both.
2. **`BufferConfig.is_sparse` is a per-buffer flag** and selects that backing. A sparse
   buffer requires a lifecycle of its own. Sparse buffers of the same size still coalesce
   with each other.
3. **Every page of a sparse buffer locks at level 0**, like every other page. Sparse pages
   are ordinary hot pages — coalesced, addressed by page index, reachable through
   `get_mem_pool_base_address`, and visible to the transceiver — that happen to live in
   host memory.
4. **All GPU residency is the per-request cache**, which is a dedicated arena rather than a
   storage tier. A sparse buffer has no separately managed GPU pages: both the block being
   written and the history a step selects occupy cache cells, drawn from the same lease and
   replaced by the same LRU.
5. **The host page is authoritative once the watermark passes the block.** A block is
   flushed to its host page by the `resize()` that raises `history_length` past it, after
   which its cache entry is evictable. Before that it holds the only copy and is pinned.
6. **Extraction is pluggable**, because KVCM2 buffers are opaque bytes and a sub-page unit
   cannot be located without layout knowledge.

## Locking and flush

`acquirePageIndex` already writes the page's *current* slot id with no level assumption
(`page.cpp:401`):

```cpp
int old = kvc->updateBasePageIndex(
    mUser.beamIndex, mUser.ordinal, mUser.lifeCycle, slotIdToPageIndexValue(pg.slotId()));
```

so `getBasePageIndices()` reports hot slot ids with no change, and because every page of a
sparse buffer locks there, the table is **uniform**: every entry is a level-0 slot id, as
for a dense buffer. Nothing resolves against the cache arena, and no caller has to know
which rule applies to which ordinal.

A block's data reaches its host page by a **flush**: a copy on the owning `KvCache`'s
stream that does **not** evict the GPU entry, which stays cached and merely stops being
pinned.

The governing rule is that **a block must be flushed before it can be selected**. Selectable
means below the watermark, so `resize()` is the deadline and must flush everything the
watermark has just passed.

`resize()` rather than `setHistoryLength()` because it is the single chokepoint: the
latter is `resize(nullopt, hist)`, `setCapacity()` is `resize(cap, nullopt)`, and a caller
may advance capacity and history together in one call. Hooking the wrapper would miss
that. Flushing inside `resize()` also composes with its rollback: a flush is a copy, so a
block flushed before a failed resize stays correctly flushed — early is always safe, and
only late is not.

`resize()` is also the *only* flush site. `commit()` needs none of its own: the manager
maintains `num_committed_tokens <= history_length <= capacity`, so a committed block is
already at or below the watermark and was flushed by the `resize()` that took the watermark
past it. Where `commit()` would advance the watermark it does so by calling `resize()`,
which flushes on the way through.

`history_length` therefore keeps two jobs: it defines selection eligibility, and it is the
flush deadline.

### Unflushed entries carry no state

An entry holds either a fetched copy of a history page or the live, still-growing block.
The first is read-only by construction — fetch only ever targets ordinals below the
watermark — and the second is written by definition. So the two populations are
distinguished by the ordinal alone:

```
unflushed  <=>  resident_ordinal >= history_length / unit_tokens
```

Nothing records this. History is monotonic (`resize()` rejects a decrease), so an ordinal
that was history stays history and a resident ordinal at or above the watermark cannot be a
stale fetched copy. The flush at `resize()` covers exactly `[old_history, new_history)`, so
the flushed prefix always equals the watermark and there is no second high-water mark to
keep. A per-entry dirty bitmap would be derived state that can only ever disagree with the
ordinal it was derived from.

Prefill and decode share one manager under in-flight batching, so the unflushed set is not
one entry: it is every in-flight block, which during prefill is the whole chunk. Two facts
keep this from being a general write-back cache:

- **Unflushed entries are private.** Only finalized pages enter the radix tree, so an
  unflushed block has exactly one owner: no aliasing, no cross-request coherence, no
  concurrent writers.
- **The unflushed set is at most one unit**, not discovered — see `max_input_length` below.
  Reservation covers it, so a fetch never meets a cache with nothing to evict.

Eviction eligibility is therefore one comparison against the watermark. There is no
evict-an-unflushed-entry path to implement: such an entry is simply not a candidate, and
the reservation guarantees enough candidates remain.

## Per-request reservation

Two separate reservations, easily confused.

**Hot pages.** Every block of the sequence has a locked hot page, so the count is
`divUp(capacity, tokens_per_block)` — the existing per-sequence locking logic, entirely
unchanged. Only the backing medium differs. This is the payoff of uniform locking: nothing
new to size, and no notion of "history pages" distinct from the rest.

**Cache cells.** `max_cached_units` is the whole cache, counting both populations:

```
cells = max_cached_units                          one buffer's cells
lease = num_buffers * divUp(cells, expansion)     contiguous cells a request takes
```

The two populations are disjoint — the selection is below the watermark, everything in
flight is above it — so the cache must hold a full selection *plus* every in-flight unit at
once. That is the only constraint, and it is what makes the second population derivable
rather than declared.

An in-flight unit's data lives only in its cache cell until the watermark passes it, so the
cache must hold every unit above the watermark. Within the supported scope that is **at most
one**: a step advances capacity one token past history, and one token straddles at most one
unit boundary.

*Sizing*: two to four times the selection width is the usual starting point. A cache only as
wide as the selection retains nothing between steps, so what it holds above the selection is
what survives to be hit next step — it is all reuse, not headroom.

### `max_input_length`

A bound on `capacity - history_length`, i.e. on how far capacity may outrun the watermark.
**One token**, and read-only.

It is a property of the supported shape rather than of how wide a cache is. A sparse request
contributes exactly one query token per step, because the fetch resolves one selection per
request while an attention kernel indexes its selection per query token; the two agree only
at one token each. This is the guard that keeps prefill and speculative decoding out — see
*Supported scope* below.

An earlier revision derived it from the reservation, as `(cells - selection_width) *
unit_tokens`, which described what the cache *could* pin rather than what a step actually
produces. Both readings reject the same accidents, but the derived one also admitted
multi-token steps whenever the cache happened to be wide, which the fetch cannot serve at
any width.

The lease therefore depends on configuration alone. It is taken once at `resume()` and
fails there if the arena is full, so the fetch path never has to handle exhaustion. Because
every request of a life cycle needs exactly the same number of cells, the arena is a
fixed-size-block free list rather than a general allocator.

`resize()` still checks the gap against this bound. It is the host-side counterpart of the
resolve kernel's in-flight overflow report: one catches a request outgrowing its cache
before the step, the other if it somehow reaches the device.

### Why the cache is a flat array of cells

A lease is one contiguous run of buffer-sized cells, and an entry resolves to one by
arithmetic alone:

```
cell    = buffer * divUp(max_cached_units, expansion) + entry / expansion
address = lease_base + cell * buffer_bytes
unit    = entry % expansion                       which unit within the cell
```

No page table, no slot indirection, no coalescing. Coalescing earns nothing inside the
cache because different layers select different units, so a slot's cells would never be
filled together; the cache took whole coalesced slots only while it drew them from the
storage allocator, which deals in slots. A dedicated arena has no such constraint, and
dropping it removes a page lookup from both the copy and the index kernel.

The consequence worth stating is that **a cache cell is not a page**. It holds one buffer's
unit, identified by `(buffer, entry index)` rather than by a block ordinal. This breaks an
assumption that runs through KVCM2 — that a page is both the unit of allocation *and* the
unit of identity — which is precisely why the cache is not a storage tier: it would have to
answer "which block does this page hold", and it has no answer.

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

KVCM2 supplies the rest, and **this is independent of the codec**. Both sides of the copy
are hot-layout: the source is a hot page, whose buffer sits at `BufferAttr::offset` within
its slot, and the destination is a cache cell holding one buffer's worth of the same
structure. The copy is therefore symmetric, one set of strides on both sides, differing only
in unit index:

```
src = hotSlotBase  + bufferOffset + c*chunkStride + srcUnit*unitStride
dst = cellBase                    + c*chunkStride + dstUnit*unitStride
```

The destination stays in natural layout rather than being repacked, so the attention kernel
sees exactly what it sees for an ordinary page.

That the codec does not enter here is the point of putting sparse history on the hot tier.
An earlier revision read *cold* pages, whose layout is the codec's concat order rather than
KVCM2's, which coupled the fetch to the codec and made compression unreachable — see
*Migration state*.

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

### The gather interface, and why there isn't one

Earlier revisions planned an `ISparseUnitGather` — a per-buffer, stream-taking device call
that would extract a unit from a representation arithmetic could not address. It existed for
exactly one reason: the fetch read cold pages, so a codec that compressed or permuted them
put the unit somewhere a layout value could not describe.

Reading hot pages removes the reason. The hot representation is KVCM2's own, always in
ordinary slot layout, and a codec by construction only transforms data on its way *out* of
that tier. A compressed level 1 is therefore invisible to the fetch, and no plugin boundary
is needed on this path at all.

This also retires the LTO-IR/nvJitLink fusion work that existed only to inline such a
gatherer back into the resolve. The extension point is gone rather than deferred, which is
worth more than the code it would have taken to build.

### Kernel structure

The descriptor path crosses **no plugin boundary**. KVCM2 resolves selections, updates LRU,
allocates cells and performs the copies itself, because a layout is data it can interpret
rather than code it must call.

It is two launches — a resolve and a copy — but that split is internal and chosen, not
forced: the resolve writes a miss list the copy consumes, which is what lets the PCIe
copies be issued before the recency bookkeeping. Fusing them would be a local optimisation,
not an interface change.

There is no second path to fuse with, so the structure is final rather than a first stage.

## Compression

Compressing history is attractive — it is the bulk of the footprint, written once and read
rarely — and with sparse history on the hot tier it needs **nothing sparse-specific**. A
codec already transforms pages crossing the hot boundary; a sparse life cycle's pages now
cross it like any other, so level 1 can hold a compressed form and the existing
`IKvCacheColdPageCodec` does the work.

The fetch is unaffected: it reads hot pages, which are never the compressed form.

`SparseHistoryPassesThroughTheCodec` pins this: a sparse life cycle's page is encoded on
demotion and decoded on promotion, with the test first asserting that the life cycle's hot
pool group really is `HOST_MEM` so that it cannot pass by accidentally describing a dense
one. Nothing in the fetch path is involved, which is the point -- it reads hot pages, which
are never the compressed form.

No compressing codec ships here, and none is needed to make the claim: compression lives
behind `IKvCacheColdPageCodec`, so "can sparse history be compressed" reduces to "does a
sparse page reach the codec", and it now does. What remains is the question the codec has
always posed and this design does not answer: whether a lossy codec is acceptable for KV
history, and under what accuracy budget.

## Fetch

One call per **layer** per step, on the caller's stream — not per layer group. A layer's
selection is computed from the previous layer's output, so a step's selections are not all
known at its start; the call has to be the unit the caller can actually fill.

Within a layer, every sparse buffer shares the call. They select the same units, so they
share one metadata row, and one resolve yields the entry list that addresses each buffer's
own cell. Their agreement is structural rather than a consequence of running identical
logic on identical input.

Models that tie a run of consecutive layers to one selection still get a call each. That is
deliberate: issuing them separately starts the first layer's copies as early as possible
instead of waiting for the whole run, and the caller simply passes the same selection array
to each call, so KVCM2 never has to know the layers are related. A caller that would rather
have the single batched launch can express it in configuration, by declaring the tied
layers' buffers as one `AttentionLayerConfig` with distinct roles — one metadata row, one
call, the same cells.

Each call then:

1. read the selection: unit ordinals, uniform width, `-1` padded;
2. build the probe index over the resident set;
3. probe once per selected unit — hit with its slot, or miss;
4. allocate a slot per miss from the free list, evicting LRU when empty;
5. extract and copy each miss from the device-mapped host page;
6. promote hits above new misses in recency order;
7. write the output index list: one index per selection slot, in the selection's own
   order, `kBadPageIndex` where a slot is padded.

The output is compact rather than a table indexed by unit ordinal, because that is the
shape every sparse attention consumer in this repository takes. A token-sparse kernel is
handed the index list in place of its page table — `attentionOp.cpp` assigns
`kvPageIdxPtr = sparse_attn_indices`, and the trtllm-gen K/V TMA descriptors are rebuilt
as a flat token array so one "page" is one token — while a block-sparse kernel takes the
list beside a page table it still uses. None indexes a table by ordinal, so producing one
would only oblige the caller to compact it again.

One index per buffer, though. A layer's buffers share the resolve but not their cells, so
the list carries a buffer dimension: `[row][buffer][selection]`, which is also the single
tensor covering K and V that attention kernels tend to want.

Steps 5 and 6 are independent. Because the probe in step 3 resolves misses directly, the
copies — the only step that touches PCIe, and so the long pole — can be issued before the
recency bookkeeping runs rather than after it.

No host involvement, so the call enqueues and returns. Every shape is static across steps.

The selection is not the only thing that needs a cell. The block being written needs one
too, and needs it before the selection takes any victim. Both are resolved by the same
probe, over one virtual list: the in-flight ordinals `[history_units, capacity_units)`
first, then the selection. An in-flight unit differs in exactly one way — a miss allocates
its cell but never joins the copy list, because its host page holds nothing to fetch until
the watermark passes it.

### Supported scope: one query token per request

The fetch is a decode-phase call. It resolves **one selection per request**, while a sparse
attention kernel indexes its selection **per query token** — trtllm-gen's
`ptrSparseMlaTopKLens` is `[numTokensQ]`, "where each tokenQ has a corresponding topK
length". The two agree only when a request contributes one query token per step, so that is
the supported shape, enforced by `max_input_length` and by `fetch_sparse_pages` requiring one
selection row per batch row.

Excluded, deliberately:

- **Prefill.** Served by a separate prefill instance under disaggregated serving, which needs
  no sparsity here: it computes the prompt and hands over the KV cache.
- **Speculative decoding and MTP.** Generation, but several query tokens per request per
  step.

Both are excluded for the same reason, and it is not the row count. The resolve makes units
*resident* in a request's cache, which holds `max_cached_units`; residency is a property of
the request, not of a query token. N query tokens each choosing their own units have a union
of up to N times the selection width, so the reservation could no longer cover a call, and a
unit evicted to make room for a later query token would invalidate an index already returned
for an earlier one. Taking one row per token would make that reachable, not fix it.

This is a difference in kind from the sparse implementations in this repository, not a gap in
degree. Their sparse prefill is real — the context path reuses the generation kernel with
per-token indices — but their indices point straight at a GPU-resident KV pool, and the
in-chunk tokens are reached by writing K/V to the cache before attention and forcing a local
window in as ordinary indices. There is nothing to stage, so no per-request cache is needed.
Here history is on host, which is exactly what makes prefill hard.

### Arriving with a history: the bulk watermark jump

A resumed request is the ordinary case under disaggregated serving, and it moves history from
zero to the whole prompt in one `resize()`. Nothing is written back, because nothing was ever
in flight: those units reached their host pages directly rather than through a cell.

The write-back is therefore bounded by the capacity the resize started from, not by how far
history travelled:

```
flush range = [old_history, min(new_history, old_capacity))    in units
```

Only a unit below the old capacity can have been in the cache at all. Without that clamp the
range is the whole prompt, which no reservation could cover, and the resize fails on a
write-back that had nothing to do.

### Capacity and eviction

Capacity is guaranteed by reservation, not back-pressure: a fetch can always satisfy its
own layer, because the reservation covers a full selection width. Within a step a fetch may
evict entries of layers already processed but never one a later kernel still needs, which
LRU recency delivers — layer N's entries are most-recently-used when layer N+1 fetches.

An entry is evictable only when its unit ordinal is **below the watermark**. That is what
protects the block being written, and it is a correctness predicate rather than a policy
hint — but it is derived, not stored: see *Unflushed entries carry no state*.

### Preconditions the caller owns

Selections must name only pure-history units; `num_blocks` carries each request's eligible
count.

Naming an ineligible one is no longer a memory-safety problem. The table is uniform, so the
ordinal resolves to a real host page of the same request — but an unflushed block's host
page has no valid contents yet, so the result is stale data rather than unrelated memory.
A debug check against the flushed state can catch it; silence would be a wrong answer, not
corruption.

## One arena, one index space

The cache arena holds nothing but cells — selected history and the unit being written alike.
That is what lets the fetch emit a **single** index list covering both, and therefore why
existing attention kernels need no change: a kernel still indexes one list against one base.

Indices are unit-granular, and over the arena rather than over a pool:

```
index = cell * expansion + unit
      = (lease_base_cell + buffer * cells_per_buffer + entry / expansion) * expansion
        + entry % expansion
```

A sparse buffer therefore has **two** base addresses, answering different questions:

| accessor | names | consumer |
| --- | --- | --- |
| `get_mem_pool_base_address` | the buffer's pages, hot tier | transceiver, KV transfer, anything addressing blocks |
| `get_sparse_cache_base_address` | the cache arena | the attention kernel, against the fetch's index list |

`get_mem_pool_base_address` keeps meaning "this buffer's pages" for every buffer, sparse or
dense, which is what lets the transceiver work unchanged on hot, coalesced pages. The arena
is a second allocation holding the only GPU-resident copy a kernel can read, so it gets a
second accessor rather than overloading the first. The consumer's side is otherwise
unchanged: one base, one index list, the expanded index space `PageIndexConverter` already
produces.

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

There is no separate budget for in-flight blocks. The sparse pool's GPU tier holds nothing
but cache cells, and an in-flight block occupies one of the same cells the selection draws
on, so `max_cached_units` covers both:

```
cache reservation = max_batch_size * N             (N in [2k, 4k])
in-flight headroom = 1 unit                        one query token per step
```

with `max_batch_size` from `constraints`. The surplus above `k` is retention, not headroom,
so `N` is sized by one demand rather than two: retention wants
`2k`–`4k`, and a prefill chunk wants enough cells to pin the whole chunk at once.

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

- **`_lockHeldBlocks`'s level hazard is gone, and so is the machinery for it.** The OOM
  rollback for `resize` re-locks at the hot level. That used to corrupt a sparse page table,
  whose entries were level-1 ids — a failure-path-only bug that survives happy-path testing.
  With every page locking at the hot level there is nothing to mismatch, so `queryLockLevel`,
  `BatchedLockTarget::cacheLevel`, `StaleBackup::cacheLevel` and `residencyLevel` have been
  removed rather than left returning a constant.
- **`_unlockStaleBlocks` is not the flush hook.** It exists for SWA blocks outside the
  window, whose content is *dead*; sparse history is live and will be read. Same trigger and
  range arithmetic, different disposition — routing sparse layers through it drops the data
  instead of flushing it. It also gates on `windowSize.has_value()`, so for a full-attention
  sparse layer it fires on nothing.
- **`_shortcutSetHistoryLength` skips the flush for exactly our configuration.** It reports
  "unchanged" when no SWA stale range moved, and for a full-attention layer
  `windowSize.has_value()` is false, so nothing is ever reported as moved: it assigns
  `mHistoryLength` and returns true, and `resize()` returns early before reaching its body.
  A flush placed in that body never runs. The shortcut must either carry the flush or be
  declined while a sparse flush is pending.
- **Hook `resize()`, not `setHistoryLength()`.** The latter is `resize(nullopt, hist)` and
  `setCapacity()` is `resize(cap, nullopt)`, so `resize()` is the only place the watermark
  moves — and a caller may move capacity and history in the same call.
- **`commit()` is not a flush site at all.** The manager maintains
  `num_committed_tokens <= history_length <= capacity`, so a committed block is already at
  or below the watermark and was flushed by the `resize()` that passed it. Where `commit()`
  advances the watermark it does so by calling `resize()`.

Also: the host pool is far larger than the GPU pool, so a page index must be widened to 64
bits before multiplying by the page stride. MSA already does this
(`triton_sparse_decode.py`); QSA and XQA should be audited.

## Migration state

The design above is the target. The code is partway there, and this section is the only
place that distinguishes the two; everything else describes the target as settled.

**Where the code stands.** Steps 1 to 3 are done: cells come from a per-life-cycle arena,
a sparse life cycle's hot pool group is host memory, `residencyLevel`'s sparse branch and
`kSparseHistoryLevel` are gone, and the fetch reads hot pages at `BufferAttr::offset`.
Compression (step 4) is unblocked but not turned on.

**Why it moved.** The codec is not the level 0 -> level 1 transform; it is the
**hot <-> cold** transform. `StorageManager::migrate` computes
`srcIsHot = srcLevel == kHotLevel` and performs a plain byte copy when source and
destination are on the same side, invoking the codec only across that boundary. History
locked at a cold level never crossed it, so no codec ever ran on it -- compression was not
deferred, it was unreachable. Two cold levels would not have helped, for the same reason.

| # | step | state |
| --- | --- | --- |
| 1 | Cells from a dedicated GPU arena, leased per request; flat cell arithmetic | **done** |
| 2 | Sparse hot pool group backed by host memory; `residencyLevel` sparse branch retired | **done** |
| 3 | Fetch reads hot pages at `BufferAttr::offset`; cold-page offset machinery gone | **done** |
| 4 | Compression, through the existing codec, with nothing sparse-specific | **reachable** |

A sparse deployment now places no requirement on its cold tiers: history is on a host-backed
*hot* pool group, so level 1 may be host, disk, or absent. An earlier revision required level
1 to be `HOST_MEM`, which was where sparse history then resided; that validation is gone.

Memory kind is a property of the **pool group**, not the level: `PoolGroupBase::cacheTier()`
is virtual, `GpuCacheLevelStorage` builds a `HostPoolGroup` for the indices it is told are
host-backed, and `migrate` dispatches the copy engine on `cacheTier(level, poolGroup)`. This
is well defined because pool-group formation already keys on `isSparse`
(`storage/config.cpp`), so a group is uniformly sparse or dense.

Step 1 deleted sub-slot addressing rather than adapting it: it existed only because the
storage allocator deals in coalesced slots, and carrying it into an arena that has none
would have preserved a workaround for a constraint that no longer exists. `SparseCacheView`
lost `cachePages` and `numPages` and gained `baseCell`; the copy and index kernels each lost
a page lookup; and the arena pre-allocates the per-request metadata, so a resume no longer
pairs a device allocation with a free.

The arena is sized from `constraints`, which already declares the batch shape `Batch` derives
its rows from. That makes the cells a static reservation rather than an on-demand draw on the
shared pool -- a change in kind, and the one the *Sizing* section assumed all along.

### Sizing a host-backed pool group

It is **off-quota**: excluded from the level's byte budget, and sized from its constraint
floor instead.

A level's quota measures one medium. Once a hot pool group is host memory it consumes none
of the device bytes that quota counts, so drawing it from there would cap sparse history at a
GPU-sized number -- the opposite of the point -- while leaving those device bytes unallocated.
Attributing it to a *host tier's* quota instead was considered and rejected: `cacheTiers` is
a positional list, so "the host tier" may be several or none, and a hot pool group's size
would then change when someone added or removed a cold tier that has nothing to do with it.

The floor is the right size because it already says what the deployment declared:
`computePoolGroupMinSlotsFromConstraints` derives it from `constraints`' capacity and scales
it by `1/maxUtilForResume`, so a deployment running at its declared shape does not trip the
resume guard. Sparse history then tracks declared context length rather than device memory.

`CacheLevelStorage::offQuotaPoolGroups()` identifies them by the only thing that matters --
a pool group whose medium differs from its level's declared tier -- so construction and the
adaptive path (`adjustCacheLevel` -> `computeSlotCountList`) get the same answer without
either being told. That mattered: excluding only at construction would have let the first
runtime rebalance undo it.

**Headroom above the floor.** A floor with no slack means every page that leaves is an
encode and every page that returns a decode, so an off-quota group is scaled above its floor
by the slack the *on-quota* groups were given:

```
headroom = clamp(quota / on_quota_min_quota, 1, K)
```

The same generosity, applied to a medium the quota does not measure. It needs no new config,
because the operator already expressed it once in the gap between `constraints` and `quota`.

`K` is clamped for two reasons, and is **provisional**. The raw ratio is driven by that gap,
and `constraints` is a floor people deliberately set low, so it reaches tens in practice; and
an over-large host allocation has no backstop the way a device allocation has the device. The
quantity that would justify a value is encode/decode volume measured against headroom, which
is not observable until a codec is configured -- so the low end of the considered range was
taken, and raising it is a one-constant change.

With no on-quota groups at all -- every attention layer sparse -- there is no ratio to take,
and the bound stands in for one.

**Coverage of the data path.** `TestSparseDataPath` asserts the bytes themselves on both
sides of the copy: a pattern written to a hot page must appear in the cell the fetch names,
and a pattern written to a cell must appear in its hot page once the watermark passes it.

This was worth building rather than arguing about. Offsetting the page base by one slot and
re-running showed **57 other sparse tests still passing** -- residency, index shape and
counts are all insensitive to which allocation the copy reads, which is exactly the mistake
the move to the hot tier could have introduced.

**A cell can never outlive its page's lock.** Moving sparse pages to the hot level made them
eligible for demotion for the first time, which raised the question of whether a page could be
encoded while the cache still held a cell copied from it. It cannot:

- An `ACTIVE` request locks the pages it requires (`AGENTS.md`, *Per-request cache*), and
  `LOCKED` forbids eviction outright (*Page status*).
- Eviction is the only route to migration, and migration across the hot boundary is the only
  thing that invokes the codec.
- The lease and the locks share a lifetime: taken at `resume()`, released together in
  `_freeReservedGpuSlots()`. A cell therefore exists only while its page is locked.

`commit()` does downgrade a lock to a holder, but only for a page it is discarding in favour
of an existing tree page -- not one the request goes on to read.

The one configuration where a live request holds unlocked pages is a **sparse layer with a
sliding window**, where `_unlockStaleBlocks` releases out-of-window blocks. It is safe, and
not by caller discipline: the indexer that produces the selection applies the same window, so
a block the window has made stale is never named. Nothing reads an unlocked page, so nothing
can read one that has been encoded or recycled.

That is worth stating because it is the *selector*, not this package, that makes it true.
`fetch_sparse_pages` cannot detect an ineligible ordinal — the page table is uniform, so a
stale ordinal still resolves to a real page of the same request. The combination is unusual
in any case: sparsity already bounds how much history a step reads.

## Settled by construction

Recorded because each looks like an open question and is not:

- **`num_blocks` lives on the device, in `Batch`.** Deriving it per call from the watermark
  would produce a host value the kernel cannot use, costing an H2D per layer.
- **Eviction never copies.** An entry is evictable only once flushed, so evicting is a
  free-list push. Flushes are triggered by the watermark alone, never by eviction pressure.
- **Pinning is not a general scheme.** The pinned set is the in-flight unit, private
  because unflushed blocks are not in the radix tree, and bounded to one by
  `max_input_length`. Ordering between the rest still comes from LRU recency alone.
- **No stale-key invalidation.** Per-request cache scope makes it impossible.
- **`tokens_per_block_override` is not a granularity lever.** See *One pool, one index
  space*.

## Resolved

- **This *is* decode-only, deliberately.** An earlier revision assumed prefill and decode
  shared one manager, and sized the in-flight headroom for a whole prefill chunk. That was
  wrong in kind, not in size: a sparse kernel indexes its selection per query token, and a
  multi-token step's selections cannot all be made resident in one per-request cache. The
  decode instance of a disaggregated deployment is the supported target; see *Supported
  scope*.
- **A selection unit never exceeds `tokens_per_block`.** Sub-page units are supported
  through `expansion`; a unit spanning several pages is rejected, which retires the
  questions of slot contiguity and k-at-a-time eviction.
- **`N` is a uniform upper bound**, not per-request. Ragged device arrays would cost the
  static shapes graph capture needs.
- **History never moves backwards.** `resize()` rejects any decrease outright
  (`kvCache.cpp`: "History length cannot be decreased"), and the only other assignment
  initialises a fresh request from a prefix match. A flushed block can therefore never be
  rewound into, with no rule of our own required.

## References

- HiSparse: [arXiv:2608.07009](https://arxiv.org/abs/2608.07009),
  [SGLang docs](https://docs.sglang.io/docs/advanced_features/hisparse_guide),
  [LMSYS blog](https://www.lmsys.org/blog/2026-04-10-sglang-hisparse/)
- MiniMax Sparse Attention: [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)
- DeepSeek-V4: [arXiv:2606.19348](https://arxiv.org/pdf/2606.19348)
