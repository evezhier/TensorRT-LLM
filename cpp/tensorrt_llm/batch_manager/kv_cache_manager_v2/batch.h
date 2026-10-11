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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/sparseCache.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

#include <cuda.h>

#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! One layer group's page table as the device sees it.
struct PageTableView
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! Per-row watermarks, in tokens. Everything a step needs about how far a request has got
//! derives from these two, at whatever granularity the consumer works in:
//!
//!     num_blocks     = historyLength / tokens_per_block     (valid page table entries)
//!     history_units  = historyLength / unit_tokens          (selectable units)
//!     capacity_units = capacity / unit_tokens               (the in-flight span's end)
//!
//! Publishing the source rather than the three derived counts keeps one truth; the divisors
//! are per-layer-group constants a kernel already has.
//!
//! A free row reads zero, so it describes a request with nothing eligible.
struct WatermarkView
{
    CUdeviceptr historyLength = 0;
    CUdeviceptr capacity = 0;
    int32_t numRows = 0;
};

// ---------------------------------------------------------------------------
// Batch — the KvCaches stepped together, and their device-side mirror.
//
// Owns the row order and the per-layer-group page tables the sparse fetch reads. Rows are
// slots, not positions in a compacted list: removing a member leaves a hole rather than
// shifting later rows, so array shapes and addresses survive a CUDA graph capture.
//
// Membership is exclusive. A KvCache detaches itself on destruction, so a dropped request
// cannot leave a dangling row.
// ---------------------------------------------------------------------------
class Batch
{
public:
    //! @param maxBatchSize    rows, i.e. the most requests that can be stepped together.
    //! @param maxBlocksPerSeq columns, i.e. the longest sequence a row can describe.
    //!
    //! Both bound device arrays whose shape must not change between steps, so neither can
    //! be grown later. Pass 0 to derive from the manager's `constraints`, which fails when
    //! none were declared -- a pool-sizing hint is not a batch bound, so it is only a
    //! default, never an inference.
    explicit Batch(KvCacheManager& manager, int maxBatchSize = 0, int maxBlocksPerSeq = 0);
    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Take the lowest free row. Throws when the batch is full, the cache already belongs to
    //! a batch, or its beam width exceeds one -- only beam 0 is mirrored, so a wider request
    //! would have every beam read beam 0's pages.
    int addKvCache(KvCache& kvCache);

    //! Free the cache's row, leaving a hole. Throws when it is not a member.
    void removeKvCache(KvCache& kvCache);

    //! Resize every member, then publish once.
    //!
    //! The sanctioned way to advance a batched step, and the reason it exists: it waits on
    //! the previous publish before touching anyone, so the staging hazard is closed by
    //! construction rather than by a caller remembering waitPublished(). Per-request
    //! operations with no Batch equivalent -- commit(), a beam fork, an SWA recycle -- still
    //! need that call, so the simple rule is to make it once at the top of a step.
    //!
    //! One result per row, in row order; false for a free row. A member that fails rolls
    //! itself back, so a batch can partially fail and the table published afterwards
    //! describes the post-rollback state.
    //!
    //! `capacities` and `historyLengths` are indexed by row, so both must be maxBatchSize
    //! long. A nullopt leaves that dimension alone, matching KvCache::resize.
    std::vector<bool> resize(std::vector<std::optional<int>> const& capacities,
        std::vector<std::optional<int>> const& historyLengths, CUstream stream);

    //! Copy the staged page indices to the device, enqueuing on `stream`.
    //!
    //! The staging rows are written by the members themselves, so they are always current;
    //! what this adds is a stream-ordered snapshot. Without it the host would be mutating
    //! the table a previously enqueued fetch is still reading.
    //!
    //! The copy is asynchronous and reads pinned memory, so it is still in flight when this
    //! returns. **Every later host write to a member's page indices must be separated from
    //! it by waitPublished()** -- a resize(), a commit(), a beam fork, an SWA recycle,
    //! anything that locks or unlocks a page. Writing sooner races the copy and publishes a
    //! torn table.
    void publish(CUstream stream);

    //! Block until the last publish() has finished reading the staging rows.
    //!
    //! Cheap once complete, and a no-op when nothing is in flight, so a caller may use it
    //! freely at the top of a step. Batch's own mutators call it; a caller manipulating a
    //! member directly is the one that has to.
    void waitPublished();

    [[nodiscard]] PageTableView pageTable(LayerGroupId layerGroupId) const;

    //! Per-row sparse cache state for one layer group, as the fetch kernel sees it.
    //!
    //! A device array of SparseCacheView, indexed by row. A free row, or a member whose
    //! cache is not reserved, reads as a default-constructed view -- null pointers and zero
    //! counts -- which the kernel skips.
    //!
    //! The views hold pointers into each request's own allocation, so what is mirrored is
    //! the handle rather than the memory: the reservation stays with the request. They move
    //! whenever a member resumes or suspends, which is why they are restaged on every
    //! publish alongside the watermarks.
    [[nodiscard]] CUdeviceptr sparseCacheViews(LayerGroupId layerGroupId) const;

    //! Per-row history length and capacity, published by the same transfer as the page
    //! tables. They have to travel together: the lengths bound which entries of a row are
    //! valid, so a kernel pairing one step's table with another step's lengths would read
    //! past the end of a row that has since been reused.
    [[nodiscard]] WatermarkView watermarks() const;

    //! Row of a member, or -1.
    [[nodiscard]] int rowOf(KvCache const& kvCache) const;

    [[nodiscard]] int maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] int maxBlocksPerSeq() const noexcept
    {
        return mMaxBlocksPerSeq;
    }

    //! Members, by row; null where a row is free.
    [[nodiscard]] std::vector<KvCache*> const& rows() const noexcept
    {
        return mRows;
    }

private:
    friend class KvCache;

    //! Called by a member being destroyed, so its row does not dangle.
    void _detach(KvCache& kvCache) noexcept;

    //! Host-side row for one (layer group, row), inside the pinned staging block.
    [[nodiscard]] int32_t* _stagingRow(LayerGroupId layerGroupId, int row) const;

    [[nodiscard]] size_t _rowElements() const noexcept;

    //! Element offset of the watermark regions within the staged block.
    [[nodiscard]] size_t _historyOffset() const noexcept;
    [[nodiscard]] size_t _capacityOffset() const noexcept;
    [[nodiscard]] size_t _totalElements() const noexcept;

    //! Refresh the staged watermarks and cache views from the members. Cheap, and it keeps
    //! KvCache ignorant of the staging: only its page indices are written through
    //! setBasePageIndexBuf.
    void _stageRowState();

    //! Byte offset of the cache view region within the staged block.
    [[nodiscard]] size_t _viewsOffsetBytes() const noexcept;
    [[nodiscard]] size_t _totalBytes() const noexcept;

    KvCacheManager* mManager = nullptr;
    int mMaxBatchSize = 0;
    int mMaxBlocksPerSeq = 0;
    LifeCycleId mNumLayerGroups{0};

    std::vector<KvCache*> mRows;

    //! Pinned so the members' index writes land here with no extra copy, and so the
    //! publish is a single asynchronous transfer per layer group.
    std::unique_ptr<HostMem> mStaging;
    CudaUniqPtr mDevice;

    //! Recorded after the publish copy, so the host can tell when the staging rows are
    //! free to write again. Null when no publish is in flight.
    CachedCudaEvent mPublished = CachedCudaEvent::makeNull();
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
