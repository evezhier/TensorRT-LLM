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

//! Status codes a resolve launch can report back through SparseResolveArgs::status.
enum class SparseResolveStatus : int32_t
{
    kOk = 0,
    //! More misses than evictable entries. Reservation is supposed to make this impossible;
    //! it means the cache was sized below the selection width plus the in-flight span.
    kOutOfEntries = 1,
    //! capacityUnits - historyUnits exceeded maxInFlightUnits, i.e. the request grew past
    //! the max_input_length it declared.
    kInFlightOverflow = 2,
    //! More units crossed the watermark in one move than maxFlushUnits allows for.
    kFlushOverflow = 3,
};

//! One layer group's resolve: turn a selection of unit ordinals into cache entry indices,
//! allocating entries for the units that are not resident.
//!
//! Layouts are [layer][...] row-major, matching SparseCacheView. Every array is device memory.
struct SparseResolveArgs
{
    //! Resident set, recency order and shape. Updated in place.
    SparseCacheView cache;

    //! [numLayers][selectionWidth] unit ordinals, negative where the selection is padded.
    //! Ordinals must be unique within a layer; duplicates would be resolved as separate
    //! misses and waste an entry each.
    int32_t const* selected = nullptr;
    //! Selected units per layer, uniform across layers and requests so shapes stay static.
    int32_t selectionWidth = 0;

    //! [1] history_length / unit_tokens, and [1] capacity / unit_tokens. Device-resident
    //! because both move every step: a captured graph must read the current value rather
    //! than one baked in at capture.
    //!
    //! Ordinals below historyUnits are selectable history. Ordinals in
    //! [historyUnits, capacityUnits) are in flight: the entry holds the only copy of their
    //! data, so it is pinned against eviction and there is nothing to fetch for it. See
    //! "Unflushed entries carry no state" in SPARSE_ATTENTION_DESIGN.md.
    int32_t const* historyUnits = nullptr;
    int32_t const* capacityUnits = nullptr;

    //! [numLayers][selectionWidth] entry index per selection, kNoSparseSlot where padded.
    int32_t* outSlot = nullptr;

    //! [numLayers][maxInFlightUnits] entry index per in-flight unit, in ordinal order from
    //! historyUnits. Entries past the live count are left untouched. The model writes new
    //! KV through these, so they are resolved before the selection takes any victim.
    int32_t* inFlightSlot = nullptr;
    //! Array stride, bounding capacityUnits - historyUnits. Derived from the declared
    //! max_input_length, so the shape stays static while the live count varies.
    int32_t maxInFlightUnits = 0;

    //! [numLayers][selectionWidth] the history units that were not resident, and the
    //! entries given to them. The copy stage reads these; only the first missCount[layer]
    //! are valid. An in-flight unit that misses is allocated an entry but never appears
    //! here: its host page holds nothing to fetch.
    int32_t* missOrdinal = nullptr;
    int32_t* missSlot = nullptr;
    //! [numLayers] number of misses.
    int32_t* missCount = nullptr;

    //! [1] SparseResolveStatus. The caller must clear it before the launch.
    int32_t* status = nullptr;
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
    //! [entriesPerLayer / expansion] GPU page slot per reserved page.
    int32_t const* cachePages = nullptr;

    //! [numLayers] offset of this layer's buffer within a cold page and within a cache
    //! page. They differ: a cold page is a concatenation in codec order.
    uint32_t const* hostBufferOffset = nullptr;
    uint32_t const* deviceBufferOffset = nullptr;

    SparseCopyDirection direction = SparseCopyDirection::kFetch;

    SparseUnitLayout layout;
    //! Selection units per buffer, i.e. tokens_per_block / tokens_per_block_override.
    int32_t expansion = 1;

    //! The work list: the resolve's misses when fetching, the flush list when writing
    //! back. [numLayers][workWidth], of which the first workCount[layer] are valid.
    int32_t const* workOrdinal = nullptr;
    int32_t const* workSlot = nullptr;
    int32_t const* workCount = nullptr;
    int32_t workWidth = 0;
    int32_t numLayers = 0;
};

//! Launch the copies. One CTA per potential work item; CTAs past a layer's count exit.
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
    //! Bound on one watermark move, which cannot exceed the declared in-flight span.
    int32_t maxFlushUnits = 0;

    //! [1] SparseResolveStatus. The caller must clear it before the launch.
    int32_t* status = nullptr;
};

//! Launch the flush-list scan. One CTA per layer.
void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream);

//! Launch the resolve. One CTA per layer; enqueues and returns.
//!
//! Throws std::invalid_argument when the shape is inconsistent or needs more shared memory
//! than the device allows.
void launchSparseResolve(SparseResolveArgs const& args, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
