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

    //! [numRows] history_length / unit_tokens, and [numRows] capacity / unit_tokens.
    //! Device-resident because both move every step: a captured graph must read the current
    //! value rather than one baked in at capture.
    //!
    //! Ordinals below historyUnits are selectable history. Ordinals in
    //! [historyUnits, capacityUnits) are in flight: the entry holds the only copy of their
    //! data, so it is pinned against eviction and there is nothing to fetch for it. See
    //! "Unflushed entries carry no state" in SPARSE_ATTENTION_DESIGN.md.
    int32_t const* historyUnits = nullptr;
    int32_t const* capacityUnits = nullptr;

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
    //! Pool holding the cold pages. Device-mapped host memory, so the kernel addresses it
    //! directly and no host round trip is needed.
    CUdeviceptr hostBase = 0;
    uint64_t hostPageStride = 0;
    //! [numBlocks] host page slot per block, indexed by block ordinal minus pageTableBase.
    //!
    //! A fetch passes the whole table and leaves the base at zero. A flush is host-initiated
    //! and touches one contiguous span of blocks, so it passes just that slice rather than
    //! requiring the full table to be device-resident.
    int32_t const* pageTable = nullptr;
    int32_t numBlocks = 0;
    int32_t pageTableBase = 0;

    //! GPU pool holding this request's reserved cache pages.
    CUdeviceptr deviceBase = 0;
    uint64_t devicePageStride = 0;
    //! [numPages] GPU page slot per reserved page.
    int32_t const* cachePages = nullptr;

    //! The reservation hands out whole coalesced slots, but the cache does not have to use
    //! them that way: a slot is buffersPerSlot buffer-sized cells, and any buffer's unit can
    //! occupy any cell. Entries are therefore addressed through a flat sub-slot index,
    //!
    //!     subSlot = buffer * subSlotsPerBuffer + entry / expansion
    //!
    //! which decouples the per-buffer partition from the page layout. Coalescing earns
    //! nothing here because layers select different units, so nothing is shared.
    int32_t subSlotsPerBuffer = 0;
    int32_t buffersPerSlot = 1;
    uint64_t bufferBytes = 0;

    //! [life cycle's buffers] offset of each buffer within a cold page, indexed by absolute
    //! buffer index. Unlike the hot side this is a lookup, not arithmetic: the cold layout
    //! is the codec's concat order.
    uint32_t const* hostBufferOffset = nullptr;

    SparseCopyDirection direction = SparseCopyDirection::kFetch;

    SparseUnitLayout layout;
    //! Selection units per buffer, i.e. tokens_per_block / tokens_per_block_override.
    int32_t expansion = 1;

    //! The work list: the resolve's misses when fetching, the flush list when writing back.
    //! [rows][workWidth], of which the first workCount[row] are valid.
    //!
    //! Its rows are layers, while the copy itself walks buffers, because a layer's buffers
    //! select the same units and so share one list. Buffer b of this call reads row
    //! `bufferLayer[bufferBase + b] - workRowBase`: a fetch resolves a single layer and
    //! passes a one-row list, a write-back crosses the watermark for every layer at once and
    //! passes them all.
    int32_t const* workOrdinal = nullptr;
    int32_t const* workSlot = nullptr;
    int32_t const* workCount = nullptr;
    int32_t workWidth = 0;

    //! [life cycle's buffers] buffer -> layer, as SparseCacheView publishes it.
    int32_t const* bufferLayer = nullptr;
    //! The buffers this call copies: `numBuffers` of them starting at `bufferBase`. They are
    //! consecutive because buffers are grouped by layer.
    int32_t bufferBase = 0;
    int32_t numBuffers = 0;
    //! Layer the work list's first row describes.
    int32_t workRowBase = 0;
};

//! Launch the copies. One CTA per (buffer, potential work item); CTAs past the buffer's
//! layer's count exit.
void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream);

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
