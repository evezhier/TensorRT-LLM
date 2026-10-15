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

    //! [numRows][selectionWidth] the history units that were not resident, and the entries
    //! given to them. The copy stage reads these for every buffer of the layer; only the
    //! first missCount[row] are valid. An in-flight unit that misses is allocated an entry but never appears
    //! here: its host page holds nothing to fetch.
    int32_t* missOrdinal = nullptr;
    int32_t* missSlot = nullptr;
    //! [numRows] number of misses.
    int32_t* missCount = nullptr;
};

//! Shared memory one CTA needs for `entriesPerLayer` entries and `selectionWidth` selections.
//! Grows as next_pow2(2 * entriesPerLayer) for the probe index, so it is the quantity that
//! bounds how large a cache a single launch can serve.
[[nodiscard]] size_t sparseResolveSharedBytes(
    int32_t entriesPerLayer, int32_t selectionWidth, int32_t maxInFlightUnits);

//! Largest entriesPerLayer this kernel can serve on `device`, given its shared memory limit.
[[nodiscard]] int32_t sparseResolveMaxEntries(int32_t selectionWidth, int32_t maxInFlightUnits, CUdevice device);

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
struct SparseUnitCopyArgs
{
    //! [numRows] per-row cache state: the lease, the per-buffer page offsets and the
    //! buffer-to-layer map all live here, because a batched fetch spans rows whose
    //! metadata belongs to different requests and so is not contiguous. A row with a null
    //! residentOrdinal holds no reservation and is skipped. A single-request write-back
    //! passes one row rather than taking a second addressing mode.
    SparseCacheView const* views = nullptr;
    int32_t numRows = 0;

    //! Pool holding the cold pages. Device-mapped host memory, so the kernel addresses it
    //! directly and no host round trip is needed.
    CUdeviceptr hostBase = 0;
    uint64_t hostPageStride = 0;
    //! [numRows][pageTableStride] host page slot per block, indexed by block ordinal minus
    //! pageTableBase.
    //!
    //! A fetch passes the batch's whole table and leaves the base at zero. A flush is
    //! host-initiated and touches one contiguous span of blocks, so it passes just that
    //! slice rather than requiring the full table to be device-resident.
    int32_t const* pageTable = nullptr;
    int32_t pageTableStride = 0;
    int32_t numBlocks = 0;
    int32_t pageTableBase = 0;

    //! Base of this call's pool region in the cell arena. One arena for every row: the lease
    //! differs per request, the arena it comes from does not. One call serves one pool,
    //! because a pool fixes the cell size.
    CUdeviceptr deviceBase = 0;

    //! A lease holds one contiguous run of buffer-sized cells in each pool's region, so an
    //! entry is addressed by arithmetic alone:
    //!
    //!     cell = views[row].leaseIndex * cellsPerLease
    //!          + views[row].bufferSlotInPool[buffer] * cellsPerBuffer
    //!          + entry / unitsPerPage
    //!
    //! No page table and no slot indirection. Coalescing would earn nothing here because
    //! different layers select different units, so a slot's cells would never fill together.
    int32_t cellsPerBuffer = 0;
    //! Cells one lease takes in this pool's region: its buffers there times cellsPerBuffer.
    int32_t cellsPerLease = 0;
    uint64_t bufferBytes = 0;

    SparseCopyDirection direction = SparseCopyDirection::kFetch;

    SparseUnitLayout layout;
    //! Selection units one page of this buffer holds.
    int32_t unitsPerPage = 1;

    //! The work list: the resolve's misses when fetching, the flush list when writing back.
    //! [numRows][workRowStride][workWidth], of which the first workCount[...] are valid.
    //!
    //! Its inner rows are layers, while the copy itself walks buffers, because a layer's
    //! buffers select the same units and so share one list. Buffer b of row r reads
    //! `(r * workRowStride + views[r].bufferLayer[bufferBase + b] - workRowBase)`.
    //!
    //! That one expression covers both callers. A fetch resolves a single layer across the
    //! batch, so it passes one list row per batch row and sets workRowBase to the layer it
    //! resolved, making the inner index zero. A write-back crosses the watermark for every
    //! layer of one request at once, so it passes one batch row and every layer.
    int32_t const* workOrdinal = nullptr;
    int32_t const* workSlot = nullptr;
    int32_t const* workCount = nullptr;
    int32_t workWidth = 0;
    int32_t workRowStride = 1;
    int32_t workRowBase = 0;

    //! The buffers this call copies: `numBuffers` of them starting at `bufferBase`. They are
    //! consecutive because buffers are grouped by layer and, within a layer, by pool.
    int32_t bufferBase = 0;
    int32_t numBuffers = 0;
};

//! Launch the copies. One CTA per (buffer, potential work item); CTAs past the buffer's
//! layer's count exit.
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
    SparseCacheView cache;

    //! [1] and [1]: the unit ordinals the watermark moved across, as [from, to). Device
    //! resident for the same reason the resolve's watermarks are.
    int32_t const* fromUnits = nullptr;
    int32_t const* toUnits = nullptr;

    //! [numLayers][maxFlushUnits] units to write back and the entries holding them.
    int32_t* outOrdinal = nullptr;
    int32_t* outSlot = nullptr;
    //! [numLayers] how many.
    int32_t* outCount = nullptr;
    //! Bound on one watermark move, which cannot exceed the in-flight headroom.
    int32_t maxFlushUnits = 0;
};

//! Launch the flush-list scan. One CTA per layer.
void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream);

//! Launch the resolve. One CTA per layer; enqueues and returns.
//!
//! Throws std::invalid_argument when the shape is inconsistent or needs more shared memory
//! than the device allows.
void launchSparseResolve(SparseResolveArgs const& args, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
