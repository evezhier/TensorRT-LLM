# Index conversion in Batch

Status: design, not implemented.

A `Batch` publishes page indices as the cache stores them -- slot ids, level-relative,
with `kBadPageIndex` preserved. Attention kernels need something else: a plain table of
indices into their own address space. Converting between the two is what this document
describes, and the point of putting it in `Batch` is that the conversion is the second half
of a transfer `Batch` already performs.

## Why it belongs here

`publish()` already has two stages: a host-to-device fill into `mDeviceStaging` on a stream
of the batch's own, then a copy into the tables kernels read, on the caller's stream. The
second stage is a plain copy only because nothing yet transforms the indices. The runtime's
dense path has the identical shape -- raw device buffer, then a conversion into the table a
captured graph reads -- so the conversion is not a new stage but the one already there,
doing work.

Folding it in removes a full pass: today the staging is copied wholesale and the runtime
separately reads its own raw table to convert it. One kernel reading the staging once
replaces both.

## The conversion

```
entry = slotId * scale + concatOffset
```

`scale` is `getPageIndexScale(layer, role)`, which resolves to the number of buffers
coalesced into that pool's page -- a slot maps to `scale` consecutive page indices, one per
coalesced buffer. `concatOffset` selects among them.

`kBadPageIndex` passes through unchanged. It has to: a sparse table numbers pure-history
and input-token entries in different levels' pools, so only the sentinel separates an absent
block from slot zero.

`concatOffset` is **derived, never configured**. A caller names the buffers that must sit in
one table; `Batch` takes the differences of their `getMemPoolBaseAddress()` values and
refuses at construction if the separation is not constant and exactly divisible. The caller
states intent; the offset is a consequence.

## Three independent axes

| axis | decided by |
|---|---|
| buffers **within** one request | always encoded in the entry -- this is the buffer dimension |
| **across layers** | the index mode |
| **across requests**, same layer | equal `scale` |

The third is what lets a quantised layer's `{K,V}` table serve `{KScale,VScale}` as well:
both pools coalesce two buffers, the pool group mirrors their slots, so the entries coincide
and one table serves both through different base pointers. Mirrored pool management is what
makes this detectable rather than accidental.

## Index mode

```
SHARED     one table per life cycle; base pointer is per layer
PER_LAYER  one table per layer;      base pointer is the shared pool-group base
```

Entries are the same formula in both. What differs is how many tables exist and where the
per-layer distinction lives -- in the base under SHARED, in the table's identity under
PER_LAYER.

Without scratch substitution one kernel covers both modes: PER_LAYER is SHARED writing to
more destinations with a different offset per layer, and a descriptor per table expresses
that uniformly. Scratch is the only thing that needs its own kernel, and only under
PER_LAYER -- it substitutes a different page for one layer, so entries stop being the affine
function of the slot id that the conversion applies.

Which means: with entries identical across the layers of a life cycle, the only thing that
can make layer *L* differ from *L+1* is **scratch-slot substitution**. That is why
`KvCache::supportsIndexMode(SHARED)` is `!hasScratchSlots()`, and why a life cycle that
cannot hold scratch -- anything that is not an `AttnLifeCycle` with a window, since
`computeScratchRange` returns empty for those -- can use SHARED even where scratch reuse is
enabled globally.

### Choosing it

Per life cycle, from the manager, at construction:

```cpp
//! Const: configuration only, never member state.
//!   PER_LAYER -> true
//!   SHARED    -> true    if this life cycle can never hold scratch
//!                nullopt if it might
std::optional<bool> KvCacheManager::supportsIndexMode(PageIndexMode, LifeCycleId) const;

Batch(..., std::optional<PageIndexMode> mode = {});
```

An explicit `mode` applies to every life cycle. Unset, each life cycle takes SHARED where
the manager answers `true` and PER_LAYER otherwise -- preferring SHARED, which is the
smaller table.

The decision is fixed at construction because a converted table's shape is baked into a
captured graph. It cannot follow a member's scratch state, which varies per step.

`KvCache::supportsIndexMode(PageIndexMode, LifeCycleId)` exists only to assert: at publish,
for each member and each SHARED life cycle, check the member holds no scratch slots there. A
violation means a request acquired scratch inside a SHARED batch, which the prefill/decode
split says cannot happen. The check is an `empty()` test per (member, SHARED life cycle), so
it stays in release builds.

## Requests and tables

A request names buffers that must be addressable from one table under one base -- a kernel
requirement, not an optimisation. Requests are per layer, since a `BufferId` is
`(layer, role)`.

```cpp
using IndexTableId = StrongIndex<int, struct IndexTableIdTag>;

//! Buffers concatenated into one table's buffer dimension, under one base.
//! Order defines the dimension.
struct IndexTableRequest { std::vector<BufferId> buffers; };

IndexTableId numIndexTables() const;
//! Every request this table satisfies. Each request's buffer order is its own mapping
//! into the buffer dimension, so K and KScale both sit at entry 0.
std::vector<IndexTableRequest> const& indexTable(IndexTableId) const;
CUdeviceptr indexTableAddress(IndexTableId) const;
//! What a kernel adds to a converted index for this buffer in this table.
MemAddress indexTableBase(IndexTableId, LayerId, DataRole) const;
//! Which table serves this buffer, and its slot in that table's buffer dimension.
std::pair<IndexTableId, int> indexTableSlot(LayerId, DataRole) const;
```

So `{{K},{V}}` on one table means K and V read identical entries through different bases.
`{{K,V},{KScale,VScale}}` means each pair is concatenated under one base, and the two pairs
share storage.

`indexTableSlot` is the normal query: a kernel asks where its indices are and at what entry,
binds the address and base, and never iterates the request list. `numIndexTables` and
`indexTable` answer the introspection question -- how many tables exist and what each covers.

Every request merged onto one table has the same buffer count; unequal sizes could not have
coinciding entries. A request spanning life cycles or scales is a hard error at
construction: no single table can serve it, and splitting silently would break the caller's
assumption that one table addresses the set.

A merged set's per-layer tables are allocated contiguously, so a caller that wants to bind
them as one strided tensor with a plane per layer can. The descriptor does not mention this,
so a caller that does not care is not obliged to.

## Buffers

| | holds | read by |
|---|---|---|
| host staging | raw, host | the upload |
| `mDeviceStaging` | raw `[lc][row][block]` | the conversion |
| `mDevice` | the converted tables | the captured kernels |

`mDevice` stays one device allocation; what changes is that it no longer mirrors the
staging's layout. It is carved into tables whose geometry comes from the mode and the merge
classes, so `indexTableAddress()` hands out offsets into it and a merged set's per-layer
tables stay contiguous. With sparse life cycles converted too, nothing captured wants the
raw form, leaving `mDeviceStaging` as the only raw copy on the device.

Batch's own device footprint rises: the converted tables carry the buffer dimension, and
under PER_LAYER the layer dimension, where the raw mirror carried neither. The system total
falls, because the converted tables move from the runtime to Batch rather than being added,
and two raw buffers stop existing -- Batch's raw mirror and the runtime's
`_device_kv_cache_block_offsets_input`.

## What this does to the sparse path

The copy reaches a cold page as `hostBase + converted * pageStride`, where `pageStride` is
one buffer and `hostBase` is the base of the index table the buffer was declared in. The
per-buffer offset the copy used to add is folded into the converted value, so
`pageBufferOffset` is gone from `SparseCacheView`, the arena's shared metadata and the cache
shape. The pool's slot size plays no part: what a converted index counts is whole buffers.
The GPU-side cache stops needing to know that a page coalesces several buffers, which was
never its concern -- it caches a buffer, not a page.

A run of buffers is copied in one launch from one base, so each buffer reads its own entry
of the table: `entry = buffer - bufferBase`, with `pageTableEntryStride` spanning the whole
table's rows. That is the same `numEntries` dimension a `{K, V}` request already produced.

Both producers of page indices read that one table, at different extents. The fetch passes
it from row zero across the batch. The write-back passes it positioned at its own row,
entries still striding by the whole table's rows -- it used to stage a private slice of raw
slot ids instead, which was a second addressing mode for the same kernel.

Two constraints follow and are checked when the Batch is built, because a run that cannot be
addressed produces wrong page addresses at step time rather than an error:

- every sparse buffer must be named by some request -- a fetch with no converted indices
  would read the wrong pages rather than fail;
- a `(layer, pool)` run must sit at consecutive entries of **one** request. Split across
  two, a single launch would need two base pointers.

That shape is not a choice -- it follows from the pool layout -- so
`KvCacheManager::sparseIndexTableRequests()` hands it out. A Batch still has to be given it:
the declaration stays explicit, and only the re-derivation is saved.

## Not in scope

Exposing the conversion as a plugin interface. The shape it would take is a two-method
interface -- one declaring what the batch must materialise, one consuming the staging -- so
that a model can substitute a lazy per-layer converter or one fused into its attention
kernel. Neither exists yet, and the default must be the safe one regardless, so the
interface waits until something needs to vary.
