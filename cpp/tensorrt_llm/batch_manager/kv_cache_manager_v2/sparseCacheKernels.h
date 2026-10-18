/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "kv_cache_manager_v2/sparseCache.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Written to an output slot that carries no selection, i.e. selection padding.
inline constexpr int32_t kNoSparseSlot = -1;

//! These kernels enforce their preconditions by trapping, not by reporting.
//!
//! Each is unreachable unless KVCM2 itself is wrong: the reservation covers a full
//! selection plus the in-flight span, and resize() rejects a gap wider than the cache can
//! pin, so a violation is a logic error rather than a runtime condition. A status word
//! would be unreadable anyway -- the trap destroys the context, so no later copy back can
//! succeed -- which is why there is none. A debug build names the condition before halting.

//! One layer's resolve across a batch: turn each row's selection of unit ordinals into
//! cache entry indices, allocating entries for the units that are not resident.
//!
//! The scope is one layer, not a layer group, because a layer's selection depends on the
//! previous layer's output: they are not all known at the start of a step. Consecutive
//! layers that happen to share a selection still get a call each, so the first layer's
//! copies start as early as possible; the caller simply passes the same selection array
//! again, and KVCM2 need not know they are related.
//!
//! One result serves every sparse buffer of the layer, since they select the same units
//! and share one metadata row. Layouts are therefore [row][...], with no buffer dimension.
//! Every array is device memory.
//! A list of units for the copy to move, compacted across rows.
//!
//! Each producer CTA claims a contiguous run with one `atomicAdd` on its list's counter and
//! writes its entries there, so the consumer walks a dense array: its grid follows the
//! machine rather than the worst case, and no warp is tied to a row that had no work. The
//! cost is that position no longer says which row an entry belongs to, so the entry says it.
//!
//! A fetch resolves one layer per launch and uses a single list, which is what `layerStride`
//! of zero means -- the layer drops out of the index. A write-back spans every layer of the
//! life cycle at once, so it takes one list per layer, chosen by the copied buffer's layer.
//!
//! Entry order is whatever the producer's CTAs raced to, which nothing depends on: the
//! entries name disjoint cells, so they may be applied in any order.
struct SparseWorkList
{
    //! [numLists][capacity] the row, the unit and the entry holding it.
    int32_t* row = nullptr;
    int32_t* ordinal = nullptr;
    int32_t* slot = nullptr;
    //! [numLists] entries written so far. Zeroed before the producer runs; the consumer
    //! bounds its loop by it, which is why the count never crosses to the host.
    int32_t* total = nullptr;

    //! Entries one list holds, and the distance between consecutive lists. Zero for a single
    //! list.
    int32_t capacity = 0;
    //! Layer the first list describes. The consumer subtracts it from a buffer's own layer.
    int32_t layerBase = 0;
    //! Lists behind `total`, so a producer can bound the layer it indexes.
    int32_t numLists = 1;
};

//! Fetch statistics, as the device counter and as its host copy.
//!
//! The two figures the resolve produces, not the two a reader wants: `hits()` is their
//! difference and is never stored. `misses` is the row's claimed run in the compacted miss
//! list -- the selected units that were not resident and so had to be copied from their
//! host pages.
//!
//! `unsigned long long` rather than `uint64_t` because the device side accumulates with
//! `atomicAdd`, which has no overload for `unsigned long` on LP64.
//!
//! 64-bit because a step at B = 256 with a 2048-wide selection resolves millions of units
//! per layer: a 32-bit pair would wrap within a few hundred steps, and a wrapped miss count
//! makes the hit rate improve.
struct SparseHitMiss
{
    //! Selection entries that named real history. Padding and in-flight units are neither
    //! hit nor miss, so neither reaches this.
    unsigned long long resolved = 0;
    //! Of those, the ones the cache did not hold.
    unsigned long long misses = 0;

    //! Clamped, because a host copy races a running fetch and the two are separate atomics:
    //! a resolve can land between them and show a miss its selection has not been counted
    //! for yet. A diagnostic must not be able to report a negative.
    [[nodiscard]] unsigned long long hits() const noexcept
    {
        return resolved >= misses ? resolved - misses : 0;
    }
};

struct SparseResolveArgs
{
    //! [numRows] cache state per row, as Batch::sparseCacheViews() publishes it. A row with
    //! a null residentOrdinal holds no reservation -- free, or a member that has suspended --
    //! and is skipped.
    SparseCacheView const* views = nullptr;
    int32_t numRows = 0;
    //! Which of the life cycle's layers this call resolves: the metadata row every sparse
    //! buffer of that layer shares.
    int32_t layerInGroup = 0;
    //! Shape shared by every row's view. Passed separately because the host needs them to
    //! size shared memory and the grid before any view can be read.
    int32_t numLayers = 0;
    int32_t entriesPerLayer = 0;

    //! [numRows][selectionWidth] unit ordinals, negative where the selection is padded.
    //! Ordinals must be unique within a row; duplicates would be resolved as separate
    //! misses and waste an entry each.
    int32_t const* selected = nullptr;
    //! Selected units per row, uniform across requests so shapes stay static.
    int32_t selectionWidth = 0;

    //! [numRows] history length and capacity, in tokens, as Batch::watermarks() publishes
    //! them; `unitTokens` divides both to units. Taking the source and the divisor rather
    //! than the two derived counts keeps one truth, and spares the caller a conversion pass
    //! over arrays it would otherwise have to allocate.
    //!
    //! Device-resident because both move every step: a captured graph must read the current
    //! value rather than one baked in at capture.
    //!
    //! Ordinals below historyUnits are selectable history. Ordinals in
    //! [historyUnits, capacityUnits) are in flight: the entry holds the only copy of their
    //! data, so it is pinned against eviction and there is nothing to fetch for it. See
    //! "Unflushed entries carry no state" in SPARSE_ATTENTION_DESIGN.md.
    int32_t const* historyTokens = nullptr;
    int32_t const* capacityTokens = nullptr;
    //! tokens_per_block / unitsPerPage, a per-layer-group constant.
    int32_t unitTokens = 1;

    //! [numRows][selectionWidth] entry index per selection, kNoSparseSlot where padded or
    //! where the row holds no reservation. One list serves every buffer of the layer.
    int32_t* outSlot = nullptr;

    //! [numRows][maxInFlightUnits] entry index per in-flight unit, in ordinal order from
    //! historyUnits. Entries past the live count are left untouched. The model writes new
    //! KV through these, so they are resolved before the selection takes any victim.
    int32_t* inFlightSlot = nullptr;
    //! Array stride, bounding capacityUnits - historyUnits. Follows from the cache width,
    //! so the shape stays static while the live count varies.
    int32_t maxInFlightUnits = 0;

    //! The history units that were not resident, and the entries given to them. One list,
    //! since a call resolves one layer; every buffer of that layer copies from it.
    //!
    //! An in-flight unit that misses is allocated an entry but never appears here: its host
    //! page is not written until the watermark passes it, so there is nothing to fetch.
    SparseWorkList work;
    //! [numRows] entries this row contributed. The copy does not read it -- it walks the
    //! compacted list -- but it is what makes a row's hit rate observable.
    int32_t* missCount = nullptr;

    //! Cumulative fetch statistics, which every row adds its own share of. Null skips it.
    //!
    //! Counted in the resolve because only the resolve can count either half: a padded
    //! selection entry is negative and an in-flight unit is not a fetch, neither of which
    //! the host can see. Both halves come from one pass, so they agree on what they count.
    //!
    //! A row adds its own contribution rather than anything reading a total afterwards. The
    //! step's counters are cleared once per publish, not per fetch, so a reader downstream
    //! would count two fetches of one layer twice.
    SparseHitMiss* hitMiss = nullptr;
};

//! Shared memory one CTA needs for `entriesPerLayer` entries and `selectionWidth` selections.
//! Grows as next_pow2(2 * entriesPerLayer) for the probe index, so it is the quantity that
//! bounds how large a cache a single launch can serve.
[[nodiscard]] size_t sparseResolveSharedBytes(
    int32_t entriesPerLayer, int32_t selectionWidth, int32_t maxInFlightUnits);

//! Largest entriesPerLayer this kernel can serve on `device`, given its shared memory limit.
[[nodiscard]] int32_t sparseResolveMaxEntries(int32_t selectionWidth, int32_t maxInFlightUnits, CUdevice device);

//! Multiprocessors `device` has, which is the one part of the grid sizing below that costs
//! a driver call and so the one part worth holding on to.
[[nodiscard]] int32_t sparseCopyMultiprocessors(CUdevice device);

//! CTAs to give a copy: enough to fill a machine of `smCount` multiprocessors, capped by
//! the most work items the list could ever hold, since beyond that the extra CTAs would
//! only exit.
//!
//! Fixed before a graph captures the copy, so it follows the machine rather than the step.
[[nodiscard]] int32_t sparseCopyGridCtas(int32_t workCapacity, int32_t smCount) noexcept;

//! Byte layout of one selection unit inside a buffer, mirroring config.h's
//! SelectionUnitLayout. Repeated here so this header stays free of the config types.
struct SparseUnitLayout
{
    uint32_t numChunks = 1;
    uint32_t chunkBytes = 0;
    uint32_t chunkStride = 0;
    uint32_t unitStride = 0;
};

//! Which way a unit moves between its cold page and its cache entry.
enum class SparseCopyDirection : uint8_t
{
    //! Host page -> cache entry, filling an entry the resolve allocated for a miss.
    kFetch = 0,
    //! Cache entry -> host page, writing back a unit the watermark has passed.
    kWriteBack = 1,
};

//! Move units between their cold pages and their cache entries.
//!
//! The default codec relocates buffer bytes verbatim, so a unit has the same internal
//! shape on both sides and one layout describes both; only the buffer's offset within a
//! page and the unit's index differ. See "Fast path: a layout value" in
//! SPARSE_ATTENTION_DESIGN.md.
//!
//! Both directions use the same addressing: the cold side is reached through the page
//! table by unit ordinal, the hot side through the cache page list by entry index. Only
//! which of the two is the source changes.
//! Vector type one pool's copy moves at a time, chosen from the layout's sizes alone.
//!
//! Sizes rather than addresses, so the width is a property of the layout and can live in a
//! descriptor built once. The two base pointers are checked against it on the host, where a
//! misaligned pool is a configuration error worth naming rather than a case to handle.
enum class SparseCopyWidth : int32_t
{
    kByte = 0,
    kWord = 1,
    kWord2 = 2,
    kWord4 = 3,
};

//! A maximal set of consecutive buffers sharing a pool, as the buffer plan groups them.
//!
//! Host-side only: it is what the manager knows about a pool, before the Batch folds each
//! buffer's own slot and index-table entry into the per-buffer descriptors below. Buffers
//! are ordered (layer, pool), so a run never spans layers.
struct SparseCopyRun
{
    int32_t bufferBase = 0;
    int32_t numBuffers = 0;
    //! Base of this pool's region in the cell arena, before a buffer's slot is added.
    CUdeviceptr deviceBase = 0;
    uint32_t bufferBytes = 0;
    int32_t cellsPerBuffer = 0;
    int32_t cellsPerLease = 0;
    uint32_t hostPageStride = 0;
    SparseUnitLayout layout;
    SparseCopyWidth width = SparseCopyWidth::kByte;
};

//! One buffer's share of a copy: everything a block needs that does not vary per work item.
//!
//! Device-resident and constant for the Batch's lifetime. Every field derives from the
//! buffer plan and the index tables, both fixed at construction, so these are built and
//! uploaded once -- which is what lets a captured graph replay a copy without any host
//! write, and what lets every buffer of every pool share one launch.
//!
//! Per buffer rather than per pool so a block resolves itself in a single load, and so the
//! two indirections a pool-level descriptor needs -- the buffer's slot within its pool and
//! its entry within the page table -- can be folded in here instead of being paid per work
//! item. `deviceBase` therefore already names this buffer's sub-region of the pool (the
//! value half of a key-and-value pool starts midway through it) and `pageTable` already
//! points at this buffer's entry.
//!
//! Sixteen-byte aligned and exactly four of them wide, so the load vectorises; the strides
//! are 32-bit because a page, a lease and a cell are all far smaller than that, and only
//! their products with a page or lease index need the wider type.
struct alignas(16) SparseCopyDesc
{
    //! Base this buffer's converted page indices are relative to. Shared by the buffers of
    //! one run: the converted index already carries each buffer's own offset within the
    //! coalesced page, which is why the pool's slot size plays no part and the stride below
    //! is one buffer.
    CUdeviceptr hostBase = 0;
    //! Converted page indices, already at this buffer's entry, so a row's run of blocks is
    //! `pageTable[row * pageTableStride ...]`. Buffers that merged onto one index table
    //! share an entry and so share this pointer; buffers of one request do not.
    int32_t const* pageTable = nullptr;
    //! This buffer's sub-region of its pool in the cell arena, which is the pool's base
    //! plus its slot there. A lease's cells follow at `leaseIndex * leaseStride`.
    CUdeviceptr deviceBase = 0;

    uint32_t hostPageStride = 0;
    uint32_t pageTableStride = 0;
    //! Blocks a row's page table describes; an ordinal past them is outside this call.
    uint32_t numBlocks = 0;
    //! Bytes one lease takes in this pool's region, which is its cells there times the
    //! cell size. Multiplied by a lease index, so widened at the multiply.
    uint32_t leaseStride = 0;
    //! Bytes of one cell, which is one buffer's worth of pages.
    uint32_t cellBytes = 0;

    SparseUnitLayout layout;

    SparseCopyWidth width = SparseCopyWidth::kByte;
};

static_assert(sizeof(SparseCopyDesc) == 64, "A copy descriptor is meant to be four vector loads");

struct SparseUnitCopyArgs
{
    //! [numRows] per-row cache state: the lease and the per-buffer pool slots live here,
    //! because a batched copy spans rows whose metadata belongs to different requests and
    //! so is not contiguous. A row with a null residentOrdinal holds no reservation.
    SparseCacheView const* views = nullptr;
    int32_t numRows = 0;

    //! One descriptor per BufferId of the life cycle, indexed by `blockIdx.y` directly.
    //! One launch for every pool: a pool's cells are disjoint from every other pool's and
    //! the converted table they share is read-only, so nothing orders them against each
    //! other, and one grid both runs them concurrently and gives every one of them the
    //! launch dependency.
    SparseCopyDesc const* copyDesc = nullptr;

    //! The buffers this call covers: `numBuffers` of them from `bufferBase`, which for a
    //! fetch is one layer's and for a write-back is the life cycle's. Consecutive, because
    //! buffers are grouped by layer and, within a layer, by pool.
    int32_t bufferBase = 0;
    int32_t numBuffers = 0;

    SparseCopyDirection direction = SparseCopyDirection::kFetch;

    //! Selection units one page holds. A life cycle's property, so every pool shares it.
    int32_t unitsPerPage = 1;

    //! [numBuffers] the layer each buffer belongs to, shared by every row of the life
    //! cycle. Read without a view because the list has to be chosen before an entry -- and
    //! so a row -- is known.
    int32_t const* bufferLayer = nullptr;

    //! The work list: the resolve's misses when fetching, the flush list when writing back.
    //! Compacted, so a warp takes the next entry rather than the next slot of a row that may
    //! hold nothing. The pools of one layer share it, because they hold the same units.
    SparseWorkList work;

    //! CTAs along the work dimension. A work item is one warp's worth -- a selection unit is
    //! 288 bytes of NVFP4 data or 36 of scales, which is 18 or 36 lanes, so a block per item
    //! would leave most of it idle -- and the warps walk the list with a grid-stride loop.
    //!
    //! Sized from the machine, not the work: the fetch is captured in a graph, so the grid
    //! is fixed before anyone knows how many units a step will miss. The device-side total
    //! is what the loop actually follows.
    int32_t gridCtas = 1;
};

//! Widest vector a pool's layout admits, from its sizes alone. A layout admitting none
//! falls to bytes, so there is never a tail to handle in the kernel.
[[nodiscard]] SparseCopyWidth sparseCopyWidth(
    SparseUnitLayout const& layout, uint64_t hostPageStride, uint64_t bufferBytes);

//! Bytes one `width` moves, for the caller that has to check its bases against it.
[[nodiscard]] size_t sparseCopyWidthBytes(SparseCopyWidth width) noexcept;

//! Launch the copies for every pool at once: the grid is `(gridCtas, numBuffers)`, where the
//! buffers of every pool form one flat range.
//!
//! One launch rather than one per pool, because nothing orders the pools against each other
//! -- their cells are disjoint and the converted table they share is read-only -- and a
//! single grid both runs them concurrently and lets every one of them carry the programmatic
//! dependency on the kernel that produced the work list.
//!
//! Enrolled in PDL when the environment enables it. The kernel resolves its pool, buffer and
//! list before waiting, so a block is resident and addressed by the time its producer drains.
void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream);

//! Turn a resolve's entry indices into the index list an attention kernel can read.
//!
//! The entry index the resolve produced is private to the cache: it names a unit within a
//! layer's partition. What a consumer needs is a value in the expanded index space,
//! `(page * buffersPerSlot + cell) * unitsPerPage + unit`.
//!
//! The result is compact -- one entry per selection slot, in the selection's own order --
//! rather than a table indexed by unit ordinal. That is the shape every sparse attention
//! consumer in this repository takes: a token-sparse kernel is handed the index list in
//! place of its page table and reads one unit per slot, and a block-sparse kernel takes
//! the list beside a page table it still uses. None indexes a table by ordinal, so
//! producing one would only oblige the caller to compact it again.
//!
//! The buffer dimension cannot be collapsed. A layer's buffers share the resolve but not
//! their cells, because the cache packs one buffer's entries densely across every cell of
//! the pages it reserved rather than leaving each at its nominal cell of a coalesced slot.
//! So one selection resolves to a different index per buffer.
struct SparseOutIndexArgs
{
    //! [numRows] per-row cache state; a row with a null residentOrdinal is filled with
    //! kBadPageIndex and nothing else, which is what a free row should look like.
    SparseCacheView const* views = nullptr;
    int32_t numRows = 0;

    //! [numRows][selectionWidth] the ordinals selected, and the entries the resolve gave
    //! them. Negative in either array means the slot carried no selection.
    int32_t const* selected = nullptr;
    int32_t const* outSlot = nullptr;
    int32_t selectionWidth = 0;

    //! Row stride of `outSlot`, when it is wider than the part this call reads. Zero means
    //! `selectionWidth`. The write-index path needs it: its slots come from the resolve's
    //! in-flight array, whose row is the reservation's headroom, while only the units
    //! actually in flight -- at most one -- are read out.
    int32_t outSlotStride = 0;

    //! [numRows][outBuffers][selectionWidth] written in full: entry `k` is the index
    //! holding `selected[row][k]` for the buffer, and kBadPageIndex where the selection is
    //! padded or the row holds no reservation. Padding stays where the selection put it
    //! rather than being packed out, so slot `k` of the output and slot `k` of the
    //! selection describe the same unit.
    //!
    //! One call fills `numBuffers` of the row's `outBuffers` entries, starting at `outBase`,
    //! because a call serves one pool while the caller asks for a whole layer.
    int32_t* out = nullptr;
    int32_t outBuffers = 0;
    int32_t outBase = 0;

    //! The buffers this call writes, as for the copy.
    int32_t bufferBase = 0;
    int32_t numBuffers = 0;

    int32_t cellsPerBuffer = 0;
    int32_t cellsPerLease = 0;
    int32_t unitsPerPage = 1;
};

//! Launch the index scatter. One CTA per (row, buffer).
void launchSparseOutIndices(SparseOutIndexArgs const& args, CUstream stream);

//! Collect the entries holding units the watermark has just passed, so they can be written
//! back to their host pages.
//!
//! Found by scanning the reverse map rather than probing: the units wanted are a contiguous
//! ordinal range, and the map has to be read in full either way.
struct SparseFlushListArgs
{
    //! [numRows] per-row cache state. A row with a null residentOrdinal holds no
    //! reservation and is skipped, as in the fetch.
    SparseCacheView const* views = nullptr;
    int32_t numRows = 0;
    //! Layers of this life cycle. The grid needs it on the host, where the views are not
    //! readable, and every row of one life cycle has the same count.
    int32_t numLayers = 0;

    //! [numRows] each: the unit ordinals that row's watermark moved across, as [from, to).
    //! Device resident for the same reason the resolve's watermarks are.
    int32_t const* fromUnits = nullptr;
    int32_t const* toUnits = nullptr;

    //! Units to write back and the entries holding them: one list per layer, because the
    //! copy that consumes it walks buffers and a buffer belongs to one layer. Batch-owned,
    //! so one launch covering every row hands the copy one pointer -- the reservation in a
    //! row's lease is its own, but this is the step's.
    SparseWorkList work;
    //! [numRows][numLayers] entries this (row, layer) contributed, for the same reason the
    //! resolve keeps its miss count: the copy does not need it, an observer does.
    int32_t* outCount = nullptr;
};

//! Launch the flush-list scan. One CTA per (layer, row).
void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream);

//! Launch the resolve. One CTA per layer; enqueues and returns.
//!
//! Throws std::invalid_argument when the shape is inconsistent or needs more shared memory
//! than the device allows.
void launchSparseResolve(SparseResolveArgs const& args, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
