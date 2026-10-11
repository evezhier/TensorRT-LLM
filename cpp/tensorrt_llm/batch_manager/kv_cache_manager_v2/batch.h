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
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

#include <cuda.h>

#include <memory>
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

    //! Take the lowest free row. Throws when the batch is full or the cache already belongs
    //! to a batch.
    int addKvCache(KvCache& kvCache);

    //! Free the cache's row, leaving a hole. Throws when it is not a member.
    void removeKvCache(KvCache& kvCache);

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
