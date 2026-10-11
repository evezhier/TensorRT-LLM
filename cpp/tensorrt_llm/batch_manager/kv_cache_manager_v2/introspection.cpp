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

#include "kv_cache_manager_v2/introspection.h"

#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <algorithm>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

bool allBlockPagesDroppable(Block const& block)
{
    for (auto const* page : block.storage)
    {
        if (page != nullptr && page->status() != PageStatus::DROPPABLE)
        {
            return false;
        }
    }

    for (auto const& [_, child] : block.next)
    {
        if (!allBlockPagesDroppable(*child))
        {
            return false;
        }
    }
    return true;
}

} // namespace

KvCacheIntrospection::ActivePageStats KvCacheIntrospection::activePageStats(KvCache const& kvCache)
{
    auto& storageMgr = kvCache.manager().storage();
    CacheLevel const numTiers = storageMgr.numCacheLevels();
    TypedVec<CacheLevel, int> counts(numTiers, 0);
    TypedVec<CacheLevel, int> unscheduledEvictable(numTiers, 0);

    for (auto const& activePage : kvCache._activePages())
    {
        auto page = kvCache._page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }

        CacheLevel const level = page->cacheLevel;
        counts.at(level) += 1;
        if (storageMgr.isEvictable(*page) && !page->scheduledForEviction())
        {
            unscheduledEvictable.at(level) += 1;
        }
    }

    return {std::move(counts), std::move(unscheduledEvictable)};
}

std::optional<bool> KvCacheIntrospection::committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId)
{
    auto page = kvCache._page(BlockOrdinal{ordinal}, kDefaultBeamIndex, LifeCycleId{lcId});
    if (!page)
    {
        return std::nullopt;
    }
    auto committed = dynamicPointerCast<CommittedPage>(page);
    if (!committed)
    {
        return std::nullopt;
    }
    return committed->block != nullptr;
}

TypedVec<PoolGroupIndex, SlotCount> KvCacheIntrospection::computeSlotsForBatch(KvCacheManager& manager,
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse)
{
    return manager.storage().computePoolGroupSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
}

std::vector<int32_t> KvCacheIntrospection::sparseCacheResidentOrdinals(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mSparseCacheState.size())
    {
        return {};
    }
    return kvCache.mSparseCacheState[id].readResidentOrdinals();
}

std::vector<int32_t> KvCacheIntrospection::sparseCachePages(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mSparseCacheState.size())
    {
        return {};
    }
    return kvCache.mSparseCacheState[id].readCachePages();
}

std::vector<int32_t> KvCacheIntrospection::sparseFlushCounts(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mSparseCacheState.size())
    {
        return {};
    }
    return kvCache.mSparseCacheState[id].readFlushCounts();
}

namespace
{

//! Device buffer holding a copy of `values`, for the resolve probe below.
CudaUniqPtr deviceCopy(std::vector<int32_t> const& values)
{
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, sizeof(int32_t) * values.size()));
    CudaUniqPtr owned{reinterpret_cast<std::byte*>(ptr)};
    cuCheck(cuMemcpyHtoD(ptr, values.data(), sizeof(int32_t) * values.size()));
    return owned;
}

CudaUniqPtr deviceCopyU32(std::vector<uint32_t> const& values)
{
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, sizeof(uint32_t) * values.size()));
    CudaUniqPtr owned{reinterpret_cast<std::byte*>(ptr)};
    cuCheck(cuMemcpyHtoD(ptr, values.data(), sizeof(uint32_t) * values.size()));
    return owned;
}

std::vector<int32_t> readBack(CudaUniqPtr const& buffer, size_t count)
{
    std::vector<int32_t> out(count);
    cuCheck(cuMemcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(buffer.get()), sizeof(int32_t) * count));
    return out;
}

} // namespace

KvCacheIntrospection::SparseResolveProbe KvCacheIntrospection::sparseResolve(
    std::vector<int32_t> const& residentOrdinal, std::vector<int32_t> const& lruOrder, int numLayers,
    int entriesPerLayer, std::vector<int32_t> const& selected, int selectionWidth, int historyUnits, int capacityUnits,
    int maxInFlightUnits)
{
    auto const entryCount = static_cast<size_t>(numLayers) * entriesPerLayer;
    auto const selectionCount = static_cast<size_t>(numLayers) * selectionWidth;
    if (residentOrdinal.size() != entryCount || lruOrder.size() != entryCount || selected.size() != selectionCount)
    {
        throw std::invalid_argument("sparseResolve: input sizes must match the declared shape");
    }

    auto resident = deviceCopy(residentOrdinal);
    auto recency = deviceCopy(lruOrder);
    auto selection = deviceCopy(selected);
    auto outSlot = deviceCopy(std::vector<int32_t>(selectionCount, 0));
    auto missOrdinal = deviceCopy(std::vector<int32_t>(selectionCount, 0));
    auto missSlot = deviceCopy(std::vector<int32_t>(selectionCount, 0));
    auto missCount = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto status = deviceCopy(std::vector<int32_t>(1, 0));
    auto history = deviceCopy(std::vector<int32_t>(1, historyUnits));
    auto capacity = deviceCopy(std::vector<int32_t>(1, capacityUnits));
    auto const inFlightCount = static_cast<size_t>(numLayers) * static_cast<size_t>(std::max(maxInFlightUnits, 0));
    auto inFlight = deviceCopy(std::vector<int32_t>(std::max<size_t>(inFlightCount, 1), kNoSparseSlot));

    SparseResolveArgs args;
    args.cache.residentOrdinal = reinterpret_cast<int32_t*>(resident.get());
    args.cache.lruOrder = reinterpret_cast<int32_t*>(recency.get());
    args.cache.numLayers = numLayers;
    args.cache.entriesPerLayer = entriesPerLayer;
    args.selected = reinterpret_cast<int32_t const*>(selection.get());
    args.selectionWidth = selectionWidth;
    args.historyUnits = reinterpret_cast<int32_t const*>(history.get());
    args.capacityUnits = reinterpret_cast<int32_t const*>(capacity.get());
    args.inFlightSlot = reinterpret_cast<int32_t*>(inFlight.get());
    args.maxInFlightUnits = maxInFlightUnits;
    args.outSlot = reinterpret_cast<int32_t*>(outSlot.get());
    args.missOrdinal = reinterpret_cast<int32_t*>(missOrdinal.get());
    args.missSlot = reinterpret_cast<int32_t*>(missSlot.get());
    args.missCount = reinterpret_cast<int32_t*>(missCount.get());
    args.status = reinterpret_cast<int32_t*>(status.get());

    launchSparseResolve(args, nullptr);
    cuCheck(cuStreamSynchronize(nullptr));

    SparseResolveProbe result;
    result.residentOrdinal = readBack(resident, entryCount);
    result.lruOrder = readBack(recency, entryCount);
    result.outSlot = readBack(outSlot, selectionCount);
    result.missOrdinal = readBack(missOrdinal, selectionCount);
    result.missSlot = readBack(missSlot, selectionCount);
    result.missCount = readBack(missCount, static_cast<size_t>(numLayers));
    result.inFlightSlot = inFlightCount > 0 ? readBack(inFlight, inFlightCount) : std::vector<int32_t>{};
    result.status = readBack(status, 1).front();
    return result;
}

std::vector<uint8_t> KvCacheIntrospection::sparseUnitCopy(std::vector<uint8_t> const& hostPool, uint64_t hostPageStride,
    std::vector<int32_t> const& pageTable, std::vector<uint8_t> const& devicePool, uint64_t devicePageStride,
    std::vector<int32_t> const& cachePages, std::vector<uint32_t> const& srcBufferOffset, int subSlotsPerLayer,
    int buffersPerSlot, uint64_t bufferBytes, uint32_t numChunks, uint32_t chunkBytes, uint32_t chunkStride,
    uint32_t unitStride, int expansion, std::vector<int32_t> const& missOrdinal, std::vector<int32_t> const& missSlot,
    std::vector<int32_t> const& missCount, int selectionWidth, int numLayers, bool writeBack)
{
    CUdeviceptr hostPtr = 0;
    cuCheck(cuMemAlloc(&hostPtr, hostPool.size()));
    CudaUniqPtr source{reinterpret_cast<std::byte*>(hostPtr)};
    cuCheck(cuMemcpyHtoD(hostPtr, hostPool.data(), hostPool.size()));

    CUdeviceptr devicePtr = 0;
    cuCheck(cuMemAlloc(&devicePtr, devicePool.size()));
    CudaUniqPtr destination{reinterpret_cast<std::byte*>(devicePtr)};
    cuCheck(cuMemcpyHtoD(devicePtr, devicePool.data(), devicePool.size()));

    auto table = deviceCopy(pageTable);
    auto pages = deviceCopy(cachePages);
    auto missOrd = deviceCopy(missOrdinal);
    auto missSlotBuf = deviceCopy(missSlot);
    auto missCountBuf = deviceCopy(missCount);
    auto srcOffsets = deviceCopyU32(srcBufferOffset);

    SparseUnitCopyArgs args;
    args.hostBase = hostPtr;
    args.hostPageStride = hostPageStride;
    args.pageTable = reinterpret_cast<int32_t const*>(table.get());
    args.numBlocks = static_cast<int32_t>(pageTable.size());
    args.deviceBase = devicePtr;
    args.devicePageStride = devicePageStride;
    args.cachePages = reinterpret_cast<int32_t const*>(pages.get());
    args.hostBufferOffset = reinterpret_cast<uint32_t const*>(srcOffsets.get());
    args.subSlotsPerLayer = subSlotsPerLayer;
    args.buffersPerSlot = buffersPerSlot;
    args.bufferBytes = bufferBytes;
    args.direction = writeBack ? SparseCopyDirection::kWriteBack : SparseCopyDirection::kFetch;
    args.layout.numChunks = numChunks;
    args.layout.chunkBytes = chunkBytes;
    args.layout.chunkStride = chunkStride;
    args.layout.unitStride = unitStride;
    args.expansion = expansion;
    args.workOrdinal = reinterpret_cast<int32_t const*>(missOrd.get());
    args.workSlot = reinterpret_cast<int32_t const*>(missSlotBuf.get());
    args.workCount = reinterpret_cast<int32_t const*>(missCountBuf.get());
    args.workWidth = selectionWidth;
    args.numLayers = numLayers;

    launchSparseUnitCopy(args, nullptr);
    cuCheck(cuStreamSynchronize(nullptr));

    // Write-back moves into the cold pool, so that is the side worth returning.
    CUdeviceptr const result = writeBack ? hostPtr : devicePtr;
    size_t const resultBytes = writeBack ? hostPool.size() : devicePool.size();
    std::vector<uint8_t> out(resultBytes);
    cuCheck(cuMemcpyDtoH(out.data(), result, resultBytes));
    return out;
}

KvCacheIntrospection::SparseFlushList KvCacheIntrospection::sparseFlushList(std::vector<int32_t> const& residentOrdinal,
    int numLayers, int entriesPerLayer, int fromUnits, int toUnits, int maxFlushUnits)
{
    auto const entryCount = static_cast<size_t>(numLayers) * entriesPerLayer;
    if (residentOrdinal.size() != entryCount)
    {
        throw std::invalid_argument("sparseFlushList: input size must match the declared shape");
    }
    auto const listCount = static_cast<size_t>(numLayers) * maxFlushUnits;

    auto resident = deviceCopy(residentOrdinal);
    auto from = deviceCopy(std::vector<int32_t>(1, fromUnits));
    auto to = deviceCopy(std::vector<int32_t>(1, toUnits));
    auto outOrdinal = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outSlot = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outCount = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto status = deviceCopy(std::vector<int32_t>(1, 0));

    SparseFlushListArgs args;
    args.cache.residentOrdinal = reinterpret_cast<int32_t*>(resident.get());
    args.cache.numLayers = numLayers;
    args.cache.entriesPerLayer = entriesPerLayer;
    args.fromUnits = reinterpret_cast<int32_t const*>(from.get());
    args.toUnits = reinterpret_cast<int32_t const*>(to.get());
    args.outOrdinal = reinterpret_cast<int32_t*>(outOrdinal.get());
    args.outSlot = reinterpret_cast<int32_t*>(outSlot.get());
    args.outCount = reinterpret_cast<int32_t*>(outCount.get());
    args.maxFlushUnits = maxFlushUnits;
    args.status = reinterpret_cast<int32_t*>(status.get());

    launchSparseFlushList(args, nullptr);
    cuCheck(cuStreamSynchronize(nullptr));

    SparseFlushList result;
    result.ordinal = readBack(outOrdinal, listCount);
    result.slot = readBack(outSlot, listCount);
    result.count = readBack(outCount, static_cast<size_t>(numLayers));
    result.status = readBack(status, 1).front();
    return result;
}

bool KvCacheIntrospection::allTreePagesDroppable(KvCacheManager& manager)
{
    for (auto const& [_, root] : manager.radixTree().roots())
    {
        for (auto const& [__, block] : root->next)
        {
            if (!allBlockPagesDroppable(*block))
            {
                return false;
            }
        }
    }
    return true;
}

void KvCacheIntrospection::setNumSampledKvCaches(KvCacheManager& manager, int value)
{
    manager.mNumSampledKvCaches = value;
}

void KvCacheIntrospection::setLastAdjustmentTime(KvCacheManager& manager, double value)
{
    manager.mLastAdjustmentTime = value;
}

void KvCacheIntrospection::setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value)
{
    manager.mTargetRatioListHot = std::move(value);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
