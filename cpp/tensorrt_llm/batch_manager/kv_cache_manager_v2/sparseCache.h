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
// A reserved slot is a row of buffer-sized cells, and any layer's unit may occupy any cell;
// a cell holds `expansion` units when the selection unit is smaller than a buffer. An entry
// is identified by (layer, entry index) and resolved to a cell arithmetically, so nothing
// ties a layer to a fixed position.
//
// The metadata is laid out [layer][entry]. It is per LAYER, not per buffer: every sparse
// buffer of a layer selects the same units, so one resident set and one recency order serve
// them all, and the entry index the resolve produces addresses each buffer's own cell. That
// makes the agreement between a layer's buffers structural rather than a consequence of
// their running identical logic on identical input.
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
    //! Sized once, because a single move cannot cross more units than the reservation's
    //! in-flight headroom.
    int32_t* flushOrdinal = nullptr;   //!< [numLayers][maxFlushUnits]
    int32_t* flushSlot = nullptr;      //!< [numLayers][maxFlushUnits]
    int32_t* flushCount = nullptr;     //!< [numLayers]
    int32_t* flushRange = nullptr;     //!< [2], the half-open unit range
    int32_t* flushPageSlice = nullptr; //!< [maxFlushBlocks]
    int32_t maxFlushUnits = 0;
    int32_t maxFlushBlocks = 0;

    //! [numBuffers] byte offset of each buffer within a cold page. A lookup rather than
    //! arithmetic: the cold layout is the codec's concat order, not this cache's choice.
    uint32_t* hostBufferOffset = nullptr;

    //! [numBuffers] the layer each buffer belongs to, i.e. which metadata row carries its
    //! residency and its work list. The copy walks buffers but reads a list produced per
    //! layer, so it needs the map to join the two. Non-decreasing, because the buffers of a
    //! layer are consecutive, which is what lets a one-layer call name its buffers as a
    //! range rather than a set.
    int32_t* bufferLayer = nullptr;

    //! Layers in this life cycle: the metadata's first dimension.
    int32_t numLayers = 0;
    //! Sparse buffers in this life cycle, one per (layer, role). Cells and cold-page offsets
    //! are per buffer, so this is the dimension the copy walks.
    int32_t numBuffers = 0;
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
    int32_t numBuffers = 0;
    int32_t entriesPerLayer = 0;
    //! GPU page slot per reserved page.
    std::vector<int32_t> cachePageSlots;
    //! [numBuffers] byte offset of each buffer within a cold page.
    std::vector<uint32_t> hostBufferOffset;
    //! [numBuffers] the layer each buffer belongs to, non-decreasing.
    std::vector<int32_t> bufferLayer;
    //! Bounds on one watermark move, from the in-flight headroom the reservation leaves.
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

    //! A device copy of view(), as a one-row array.
    //!
    //! The copy kernel reads its per-row state through a view array, so that a batched
    //! fetch can serve rows whose metadata lives in different requests' allocations. A
    //! single-request write-back takes the same path with one row rather than a second
    //! addressing mode. Fixed for the life of the reservation, like everything it names.
    [[nodiscard]] SparseCacheView const* deviceView() const noexcept
    {
        return mDeviceView;
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
    SparseCacheView* mDeviceView = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
