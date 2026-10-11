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

#include "kv_cache_manager_v2/batch.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Largest batch and longest sequence the declared constraints describe, or zero when none
//! were declared.
std::pair<int, int> deriveBounds(KVCacheManagerConfig const& config)
{
    int rows = 0;
    int capacity = 0;
    for (auto const& desc : config.constraints)
    {
        rows = std::max(rows, static_cast<int>(desc.kvCaches.size()));
        for (auto const& kvCache : desc.kvCaches)
        {
            capacity = std::max(capacity, kvCache.capacity);
        }
    }
    return {rows, capacity == 0 ? 0 : divUp(capacity, config.tokensPerBlock)};
}

} // namespace

Batch::Batch(KvCacheManager& manager, int maxBatchSize, int maxBlocksPerSeq)
    : mManager(&manager)
{
    auto const [derivedRows, derivedBlocks] = deriveBounds(manager.config());
    mMaxBatchSize = maxBatchSize > 0 ? maxBatchSize : derivedRows;
    mMaxBlocksPerSeq = maxBlocksPerSeq > 0 ? maxBlocksPerSeq : derivedBlocks;
    if (mMaxBatchSize <= 0 || mMaxBlocksPerSeq <= 0)
    {
        throw std::invalid_argument(
            "Batch needs a row and block bound; pass them explicitly when the manager declares no constraints");
    }

    mNumLayerGroups = manager.storage().numLifeCycles();
    mRows.assign(static_cast<size_t>(mMaxBatchSize), nullptr);

    auto const elements = _rowElements() * static_cast<size_t>(mNumLayerGroups.value());
    mStaging = std::make_unique<HostMem>(sizeof(int32_t) * elements);
    auto* staging = reinterpret_cast<int32_t*>(mStaging->address());
    // A row that no member occupies must still read as "no such block" rather than as slot
    // zero, since the fetch has no other way to tell them apart.
    std::fill(staging, staging + elements, kBadPageIndex.value());

    CUdeviceptr device = 0;
    cuCheck(cuMemAlloc(&device, sizeof(int32_t) * elements));
    mDevice = CudaUniqPtr{reinterpret_cast<std::byte*>(device)};
    cuCheck(cuMemcpyHtoD(device, staging, sizeof(int32_t) * elements));
}

Batch::~Batch()
{
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            for (auto* kvCache : mRows)
            {
                if (kvCache != nullptr)
                {
                    removeKvCache(*kvCache);
                }
            }
        });
}

size_t Batch::_rowElements() const noexcept
{
    return static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
}

int32_t* Batch::_stagingRow(LayerGroupId layerGroupId, int row) const
{
    auto* base = reinterpret_cast<int32_t*>(mStaging->address());
    auto const offset
        = _rowElements() * static_cast<size_t>(layerGroupId.value()) + static_cast<size_t>(row) * mMaxBlocksPerSeq;
    return base + offset;
}

int Batch::addKvCache(KvCache& kvCache)
{
    if (kvCache.mBatch != nullptr)
    {
        throw LogicError("A KvCache belongs to at most one Batch");
    }
    // setBasePageIndexBuf writes this row, so the last publish must be done reading it.
    waitPublished();
    auto const free = std::find(mRows.begin(), mRows.end(), nullptr);
    if (free == mRows.end())
    {
        throw LogicError("Batch is full; it was sized for max_batch_size rows");
    }
    auto const row = static_cast<int>(std::distance(mRows.begin(), free));

    // Point every layer group's index table at this row. setBasePageIndexBuf copies the
    // current indices in and keeps writing there, so the staging row needs no refresh of
    // its own afterwards.
    for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
    {
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lg, _stagingRow(lg, row), mMaxBlocksPerSeq);
    }

    *free = &kvCache;
    kvCache.mBatch = this;
    kvCache.mBatchRow = row;
    return row;
}

void Batch::removeKvCache(KvCache& kvCache)
{
    if (kvCache.mBatch != this)
    {
        throw LogicError("KvCache is not a member of this Batch");
    }
    waitPublished();
    int const row = kvCache.mBatchRow;
    for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
    {
        // Hands the indices back to the cache's own storage before the row is reused.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lg, nullptr, 0);
        auto* staged = _stagingRow(lg, row);
        std::fill(staged, staged + mMaxBlocksPerSeq, kBadPageIndex.value());
    }
    mRows[static_cast<size_t>(row)] = nullptr;
    kvCache.mBatch = nullptr;
    kvCache.mBatchRow = -1;
}

void Batch::_detach(KvCache& kvCache) noexcept
{
    if (kvCache.mBatch != this || kvCache.mBatchRow < 0)
    {
        return;
    }
    KVCM2_POISON_ON_EXCEPT([this] { waitPublished(); });
    int const row = kvCache.mBatchRow;
    for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
    {
        auto* staged = _stagingRow(lg, row);
        std::fill(staged, staged + mMaxBlocksPerSeq, kBadPageIndex.value());
    }
    mRows[static_cast<size_t>(row)] = nullptr;
    kvCache.mBatch = nullptr;
    kvCache.mBatchRow = -1;
}

std::vector<bool> Batch::resize(std::vector<std::optional<int>> const& capacities,
    std::vector<std::optional<int>> const& historyLengths, CUstream stream)
{
    if (capacities.size() != mRows.size() || historyLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch::resize: one capacity and history length per row");
    }
    // Before the first member is touched, since each resize writes its staging row.
    waitPublished();

    std::vector<bool> results(mRows.size(), false);
    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto* kvCache = mRows[row];
        if (kvCache == nullptr)
        {
            continue;
        }
        results[row] = kvCache->resize(capacities[row], historyLengths[row]);
    }
    // Once, after every member has settled: a partial failure leaves the rolled-back state,
    // which is what the table should describe.
    publish(stream);
    return results;
}

void Batch::publish(CUstream stream)
{
    // A publish already in flight is reading the same staging rows this one will, and the
    // copies are ordered on the stream anyway; waiting keeps the event meaning "the staging
    // rows are free" rather than "the last of several copies finished".
    waitPublished();

    // Whole-table for now. Refreshing only the rows whose locks moved is an optimisation
    // the design allows for: it degrades into a slower step, never a wrong answer.
    auto const bytes = sizeof(int32_t) * _rowElements() * static_cast<size_t>(mNumLayerGroups.value());
    cuCheck(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(mDevice.get()), reinterpret_cast<void*>(mStaging->address()), bytes, stream));
    mPublished = CachedCudaEvent{reinterpret_cast<CudaStream>(stream)};
}

void Batch::waitPublished()
{
    if (mPublished.isClosed())
    {
        return;
    }
    mPublished.synchronize();
    // Closing returns the event to the pool and makes every later call free.
    mPublished.close();
}

PageTableView Batch::pageTable(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::pageTable: unknown layer group");
    }
    PageTableView view;
    view.data = reinterpret_cast<CUdeviceptr>(mDevice.get())
        + sizeof(int32_t) * _rowElements() * static_cast<size_t>(layerGroupId.value());
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

int Batch::rowOf(KvCache const& kvCache) const
{
    return kvCache.mBatch == this ? kvCache.mBatchRow : -1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
