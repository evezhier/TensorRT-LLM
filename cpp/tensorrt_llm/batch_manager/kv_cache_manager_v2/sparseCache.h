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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/stagingBuffer.h"

#include <cstdint>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SparseCacheView — the device-resident state of one request's sparse history
// cache for one life cycle, as the fetch kernel sees it.
//
// A reserved page carries one buffer per coalesced layer, so a page yields one entry per
// layer (or `expansion` entries when the selection unit is smaller than a page). An entry
// is therefore identified by (layer, entry index), not by page: one page may hold
// different layers' data for entirely different ordinals.
//
// Both arrays are laid out [layer][entry], row-major over entriesPerLayer.
// ---------------------------------------------------------------------------
struct SparseCacheView
{
    //! Resident unit ordinal per entry, kBadBlockOrdinal when the entry is free.
    //!
    //! Dense rather than keyed: answering "is this ordinal resident, and where" is done by
    //! building a hash over this array in shared memory at the top of each fetch, so the
    //! persistent form stays proportional to cache capacity rather than to context length.
    int32_t* residentOrdinal = nullptr;

    //! Entry indices in recency order, most-recently-used first.
    int32_t* lruOrder = nullptr;

    //! [numPages] GPU page slot per reserved cache page, so the copy can turn an entry
    //! index into an address. Fixed for the life of the reservation.
    int32_t* cachePages = nullptr;
    int32_t numPages = 0;

    //! Scratch for one write-back: the units a watermark move crossed and the cells holding
    //! them, the watermark scalars the scan reads, the page slots of the blocks touched, and
    //! a status word. Sized once, because a single move cannot cross more units than the
    //! declared in-flight span.
    int32_t* flushOrdinal = nullptr;   //!< [numLayers][maxFlushUnits]
    int32_t* flushSlot = nullptr;      //!< [numLayers][maxFlushUnits]
    int32_t* flushCount = nullptr;     //!< [numLayers]
    int32_t* flushRange = nullptr;     //!< [2], the half-open unit range
    int32_t* flushPageSlice = nullptr; //!< [maxFlushBlocks]
    int32_t* status = nullptr;         //!< [1] SparseResolveStatus
    int32_t maxFlushUnits = 0;
    int32_t maxFlushBlocks = 0;

    //! [numLayers] byte offset of each buffer within a cold page. A lookup rather than
    //! arithmetic: the cold layout is the codec's concat order, not this cache's choice.
    uint32_t* hostBufferOffset = nullptr;

    int32_t numLayers = 0;
    int32_t entriesPerLayer = 0;

    [[nodiscard]] constexpr int32_t numEntries() const noexcept
    {
        return numLayers * entriesPerLayer;
    }
};

// ---------------------------------------------------------------------------
// SparseCacheInit — everything a request's cache state is built from, all of it
// fixed for the life of the reservation.
// ---------------------------------------------------------------------------
struct SparseCacheInit
{
    int32_t numLayers = 0;
    int32_t entriesPerLayer = 0;
    //! GPU page slot per reserved page.
    std::vector<int32_t> cachePageSlots;
    //! [numLayers] byte offset of each buffer within a cold page.
    std::vector<uint32_t> hostBufferOffset;
    //! Bounds on one watermark move, from the declared max_input_length.
    int32_t maxFlushUnits = 0;
    int32_t maxFlushBlocks = 0;
};

// ---------------------------------------------------------------------------
// SparseCacheState — owns the device memory behind a SparseCacheView.
//
// One allocation holds every array; view() hands out the carved pointers. Created
// when a request reserves its cache pages and destroyed when it releases them, so the
// metadata's lifetime matches the pages it describes.
// ---------------------------------------------------------------------------
class SparseCacheState
{
public:
    SparseCacheState() = default;

    //! Allocate and initialise state for `numLayers` layers of `entriesPerLayer` entries,
    //! backed by the GPU page slots in `cachePageSlots`.
    //!
    //! Entries start free: every ordinal is kBadBlockOrdinal and the recency order is the
    //! identity permutation. Initialisation is enqueued on `stream`.
    explicit SparseCacheState(SparseCacheInit const& init, CUstream stream);

    [[nodiscard]] SparseCacheView const& view() const noexcept
    {
        return mView;
    }

    [[nodiscard]] bool empty() const noexcept
    {
        return mMemory == nullptr;
    }

    //! Copy the resident ordinals back to the host. Test and debug use only; synchronises.
    [[nodiscard]] std::vector<int32_t> readResidentOrdinals() const;

    //! Copy the per-buffer write-back counts back to the host, -1 where no write-back has
    //! run. Test and debug use only; synchronises.
    [[nodiscard]] std::vector<int32_t> readFlushCounts() const;

    //! Copy the cache page slots back to the host. Test and debug use only; synchronises.
    [[nodiscard]] std::vector<int32_t> readCachePages() const;

private:
    CudaUniqPtr mMemory;
    SparseCacheView mView;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
