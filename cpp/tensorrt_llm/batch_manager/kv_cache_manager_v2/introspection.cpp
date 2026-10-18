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

#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <algorithm>
#include <numeric>
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
    if (id >= kvCache.mSparseCache.size())
    {
        return {};
    }
    return kvCache.mSparseCache[id].readResidentOrdinals();
}

int KvCacheIntrospection::sparseCacheBaseCell(KvCache const& kvCache, int lcId, int pool)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mSparseCache.size() || kvCache.mSparseCache[id].empty())
    {
        return -1;
    }
    // Relative to the pool's own arena region, which is what getSparseCacheBaseAddress
    // reports for a buffer drawing on it.
    auto const& arena = kvCache.mManager->mSparseArenas[id];
    return kvCache.mSparseCache[id].view().leaseIndex * arena.cellsPerLease(PoolIndex{pool});
}

std::vector<int32_t> KvCacheIntrospection::batchPageTable(Batch const& batch, int layerGroupId)
{
    auto const view = batch.pageTable(LayerGroupId{layerGroupId});
    std::vector<int32_t> out(static_cast<size_t>(view.numRows) * view.blocksPerRow);
    cuCheck(cuMemcpyDtoH(out.data(), view.data, sizeof(int32_t) * out.size()));
    return out;
}

std::vector<int32_t> KvCacheIntrospection::batchIndexTable(Batch const& batch, int tableId)
{
    IndexTableId const typed{tableId};
    auto const entries = static_cast<size_t>(batch.indexTable(typed).front().buffers.size());
    auto const perEntry = static_cast<size_t>(batch.maxBatchSize()) * static_cast<size_t>(batch.maxBlocksPerSeq());
    std::vector<int32_t> out(entries * perEntry);
    cuCheck(cuMemcpyDtoH(out.data(), batch.indexTableAddress(typed), sizeof(int32_t) * out.size()));
    return out;
}

std::vector<int32_t> KvCacheIntrospection::batchSparseViewEntries(Batch const& batch, int layerGroupId)
{
    auto const rows = static_cast<size_t>(batch.maxBatchSize());
    std::vector<SparseCacheView> views(rows);
    cuCheck(
        cuMemcpyDtoH(views.data(), batch.sparseCacheViews(LayerGroupId{layerGroupId}), sizeof(SparseCacheView) * rows));

    std::vector<int32_t> out;
    out.reserve(rows);
    for (auto const& view : views)
    {
        out.push_back(view.residentOrdinal == nullptr ? 0 : view.entriesPerLayer);
    }
    return out;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> KvCacheIntrospection::batchWatermarks(Batch const& batch)
{
    auto const view = batch.watermarks();
    std::vector<int32_t> history(static_cast<size_t>(view.numRows));
    std::vector<int32_t> capacity(static_cast<size_t>(view.numRows));
    cuCheck(cuMemcpyDtoH(history.data(), view.historyLength, sizeof(int32_t) * history.size()));
    cuCheck(cuMemcpyDtoH(capacity.data(), view.capacity, sizeof(int32_t) * capacity.size()));
    return {std::move(history), std::move(capacity)};
}

std::vector<int32_t> KvCacheIntrospection::sparseFlushCounts(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mSparseCache.size() || kvCache.mBatch == nullptr)
    {
        return {};
    }
    // The lists are the step's, not the request's, so this reads the Batch's row for it.
    auto const& scratch = kvCache.mBatch->sparseStepScratch(LayerGroupId{lcId});
    auto const layers = static_cast<size_t>(scratch.numLayers);
    std::vector<int32_t> out(layers);
    cuCheck(cuMemcpyDtoH(out.data(),
        reinterpret_cast<CUdeviceptr>(scratch.flushCount + static_cast<size_t>(kvCache.mBatchRow) * layers),
        sizeof(int32_t) * layers));
    return out;
}

namespace
{

//! Device buffer holding a copy of `values`, for the resolve probe below.
CudaUniqPtr<> deviceCopy(std::vector<int32_t> const& values)
{
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, sizeof(int32_t) * values.size()));
    CudaUniqPtr<> owned{reinterpret_cast<std::byte*>(ptr)};
    cuCheck(cuMemcpyHtoD(ptr, values.data(), sizeof(int32_t) * values.size()));
    return owned;
}

CudaUniqPtr<> deviceCopyU32(std::vector<uint32_t> const& values)
{
    CUdeviceptr ptr = 0;
    cuCheck(cuMemAlloc(&ptr, sizeof(uint32_t) * values.size()));
    CudaUniqPtr<> owned{reinterpret_cast<std::byte*>(ptr)};
    cuCheck(cuMemcpyHtoD(ptr, values.data(), sizeof(uint32_t) * values.size()));
    return owned;
}

std::vector<int32_t> readBack(CudaUniqPtr<> const& buffer, size_t count)
{
    std::vector<int32_t> out(count);
    cuCheck(cuMemcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(buffer.get()), sizeof(int32_t) * count));
    return out;
}

} // namespace

KvCacheIntrospection::SparseResolveProbe KvCacheIntrospection::sparseResolve(
    std::vector<int32_t> const& residentOrdinal, std::vector<int32_t> const& lruOrder, int numLayers,
    int entriesPerLayer, std::vector<int32_t> const& selected, int selectionWidth,
    std::vector<int32_t> const& historyUnits, std::vector<int32_t> const& capacityUnits, int maxInFlightUnits,
    std::vector<int32_t> const& inactiveRows)
{
    auto const numRows = static_cast<int>(historyUnits.size());
    auto const perRowEntries = static_cast<size_t>(numLayers) * entriesPerLayer;
    auto const entryCount = perRowEntries * static_cast<size_t>(numRows);
    auto const perRowSelection = static_cast<size_t>(numLayers) * selectionWidth;
    auto const selectionCount = perRowSelection * static_cast<size_t>(numRows);
    if (numRows == 0 || capacityUnits.size() != historyUnits.size() || residentOrdinal.size() != entryCount
        || lruOrder.size() != entryCount || selected.size() != selectionCount)
    {
        throw std::invalid_argument("sparseResolve: input sizes must match the declared shape");
    }

    auto resident = deviceCopy(residentOrdinal);
    auto recency = deviceCopy(lruOrder);
    auto selection = deviceCopy(selected);
    auto outSlot = deviceCopy(std::vector<int32_t>(selectionCount, 0));
    auto missRow = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missOrdinal = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missSlot = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missTotal = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto const countCount = static_cast<size_t>(numLayers) * static_cast<size_t>(numRows);
    auto missCount = deviceCopy(std::vector<int32_t>(countCount, 0));
    auto history = deviceCopy(historyUnits);
    auto capacity = deviceCopy(capacityUnits);
    auto const inFlightCount = countCount * static_cast<size_t>(std::max(maxInFlightUnits, 0));
    auto inFlight = deviceCopy(std::vector<int32_t>(std::max<size_t>(inFlightCount, 1), kNoSparseSlot));

    // One view per row, each pointing at that row's slice of the two shared allocations.
    // Mirrors what Batch stages: the handle is per row, the memory stays with the request.
    std::vector<SparseCacheView> views(static_cast<size_t>(numRows));
    for (int row = 0; row < numRows; ++row)
    {
        auto& view = views[static_cast<size_t>(row)];
        if (std::find(inactiveRows.begin(), inactiveRows.end(), row) != inactiveRows.end())
        {
            // Left default: a row holding no reservation, which the kernel must skip.
            continue;
        }
        view.residentOrdinal = reinterpret_cast<int32_t*>(resident.get()) + perRowEntries * row;
        view.lruOrder = reinterpret_cast<int32_t*>(recency.get()) + perRowEntries * row;
        view.numLayers = numLayers;
        view.entriesPerLayer = entriesPerLayer;
    }
    CUdeviceptr viewPtr = 0;
    cuCheck(cuMemAlloc(&viewPtr, sizeof(SparseCacheView) * views.size()));
    CudaUniqPtr<> viewOwner{reinterpret_cast<std::byte*>(viewPtr)};
    cuCheck(cuMemcpyHtoD(viewPtr, views.data(), sizeof(SparseCacheView) * views.size()));

    // One launch per layer, as a step does: a layer's selection is not known until the
    // previous layer has run. The arrays are therefore layer-major, so each launch sees a
    // contiguous [row][...] slice of its own. Same stream, so the layers stay ordered.
    for (int layer = 0; layer < numLayers; ++layer)
    {
        auto const selectionOffset = static_cast<size_t>(layer) * numRows * selectionWidth;
        auto const countOffset = static_cast<size_t>(layer) * numRows;

        SparseResolveArgs args;
        args.views = reinterpret_cast<SparseCacheView const*>(viewPtr);
        args.numRows = numRows;
        args.layerInGroup = layer;
        args.numLayers = numLayers;
        args.entriesPerLayer = entriesPerLayer;
        args.selected = reinterpret_cast<int32_t const*>(selection.get()) + selectionOffset;
        args.selectionWidth = selectionWidth;
        // The hook works in units, so one token is one unit.
        args.historyTokens = reinterpret_cast<int32_t const*>(history.get());
        args.capacityTokens = reinterpret_cast<int32_t const*>(capacity.get());
        args.unitTokens = 1;
        args.inFlightSlot
            = reinterpret_cast<int32_t*>(inFlight.get()) + countOffset * static_cast<size_t>(maxInFlightUnits);
        args.maxInFlightUnits = maxInFlightUnits;
        args.outSlot = reinterpret_cast<int32_t*>(outSlot.get()) + selectionOffset;
        // One list per layer, each the width of the whole batch's selection: the worst case
        // is every row missing everything it asked for.
        args.work.row = reinterpret_cast<int32_t*>(missRow.get()) + selectionOffset;
        args.work.ordinal = reinterpret_cast<int32_t*>(missOrdinal.get()) + selectionOffset;
        args.work.slot = reinterpret_cast<int32_t*>(missSlot.get()) + selectionOffset;
        args.work.total = reinterpret_cast<int32_t*>(missTotal.get()) + layer;
        args.work.capacity = numRows * selectionWidth;
        args.work.numLists = 1;
        args.missCount = reinterpret_cast<int32_t*>(missCount.get()) + countOffset;

        launchSparseResolve(args, nullptr);
    }
    cuCheck(cuStreamSynchronize(nullptr));

    SparseResolveProbe result;
    result.residentOrdinal = readBack(resident, entryCount);
    result.lruOrder = readBack(recency, entryCount);
    result.outSlot = readBack(outSlot, selectionCount);
    result.missRow = readBack(missRow, selectionCount);
    result.missOrdinal = readBack(missOrdinal, selectionCount);
    result.missSlot = readBack(missSlot, selectionCount);
    result.missTotal = readBack(missTotal, static_cast<size_t>(numLayers));
    result.missCount = readBack(missCount, countCount);
    result.inFlightSlot = inFlightCount > 0 ? readBack(inFlight, inFlightCount) : std::vector<int32_t>{};
    return result;
}

std::vector<uint8_t> KvCacheIntrospection::sparseUnitCopy(std::vector<uint8_t> const& hostPool, uint64_t hostPageStride,
    std::vector<int32_t> const& pageTable, std::vector<uint8_t> const& deviceArena, int baseCell, int numBlocks,
    int cellsPerBuffer, uint64_t bufferBytes, uint32_t numChunks, uint32_t chunkBytes, uint32_t chunkStride,
    uint32_t unitStride, int unitsPerPage, std::vector<int32_t> const& missOrdinal,
    std::vector<int32_t> const& missSlot, std::vector<int32_t> const& missCount, int selectionWidth,
    std::vector<int32_t> const& bufferLayer, bool writeBack, int gridCtas)
{
    CUdeviceptr hostPtr = 0;
    cuCheck(cuMemAlloc(&hostPtr, hostPool.size()));
    CudaUniqPtr<> source{reinterpret_cast<std::byte*>(hostPtr)};
    cuCheck(cuMemcpyHtoD(hostPtr, hostPool.data(), hostPool.size()));

    CUdeviceptr devicePtr = 0;
    cuCheck(cuMemAlloc(&devicePtr, deviceArena.size()));
    CudaUniqPtr<> destination{reinterpret_cast<std::byte*>(devicePtr)};
    cuCheck(cuMemcpyHtoD(devicePtr, deviceArena.data(), deviceArena.size()));

    auto table = deviceCopy(pageTable);
    // The caller still describes work as [list][selectionWidth] with a count per list,
    // which is the shape a test can write by hand. Compact it here, because that is what
    // the kernel consumes: a dense run per list, with the row carried by the entry.
    auto const numLists = bufferLayer.empty() ? 1 : bufferLayer.back() + 1;
    auto const perList = static_cast<size_t>(std::max(selectionWidth, 1));
    std::vector<int32_t> flatRow;
    std::vector<int32_t> flatOrdinal;
    std::vector<int32_t> flatSlot;
    std::vector<int32_t> totals(static_cast<size_t>(numLists), 0);
    flatRow.reserve(missOrdinal.size());
    flatOrdinal.reserve(missOrdinal.size());
    flatSlot.reserve(missOrdinal.size());
    for (int list = 0; list < numLists; ++list)
    {
        auto const count = static_cast<size_t>(missCount.at(static_cast<size_t>(list)));
        for (size_t j = 0; j < perList; ++j)
        {
            auto const index = static_cast<size_t>(list) * perList + j;
            // The hook drives one request, so every entry names row zero.
            flatRow.push_back(0);
            flatOrdinal.push_back(j < count ? missOrdinal.at(index) : kNoSparseSlot);
            flatSlot.push_back(j < count ? missSlot.at(index) : kNoSparseSlot);
        }
        totals[static_cast<size_t>(list)] = static_cast<int32_t>(count);
    }

    auto workRowBuf = deviceCopy(flatRow);
    auto missOrd = deviceCopy(flatOrdinal);
    auto missSlotBuf = deviceCopy(flatSlot);
    auto totalBuf = deviceCopy(totals);
    auto bufferLayerBuf = deviceCopy(bufferLayer);

    // The kernel reads the lease and the per-buffer maps through a view, so the hook
    // assembles one standing in for a single request's cache state.
    SparseCacheView view;
    // Non-null only to mark the row live; the copy itself never reads the metadata arrays.
    view.residentOrdinal = reinterpret_cast<int32_t*>(table.get());
    // The probe drives one pool, so a lease of one cell and a buffer-slot identity reproduce
    // the address the production path computes from a lease index and a pool slot.
    view.leaseIndex = baseCell;
    view.bufferLayer = reinterpret_cast<int32_t*>(bufferLayerBuf.get());
    std::vector<int32_t> slotInPool(bufferLayer.size());
    std::iota(slotInPool.begin(), slotInPool.end(), 0);
    auto slotInPoolBuf = deviceCopy(slotInPool);
    view.bufferSlotInPool = reinterpret_cast<int32_t*>(slotInPoolBuf.get());
    CUdeviceptr viewPtr = 0;
    cuCheck(cuMemAlloc(&viewPtr, sizeof(SparseCacheView)));
    CudaUniqPtr<> viewOwner{reinterpret_cast<std::byte*>(viewPtr)};
    cuCheck(cuMemcpyHtoD(viewPtr, &view, sizeof(SparseCacheView)));

    // One descriptor per buffer, as the real path builds them: the hook drives the
    // addressing, so every buffer shares a pool and differs only by its slot in it.
    std::vector<SparseCopyDesc> descs(bufferLayer.size());
    for (size_t i = 0; i < descs.size(); ++i)
    {
        auto& desc = descs[i];
        desc.hostBase = hostPtr;
        desc.hostPageStride = static_cast<uint32_t>(hostPageStride);
        desc.pageTable = reinterpret_cast<int32_t const*>(table.get()) + i * static_cast<size_t>(numBlocks);
        desc.pageTableStride = static_cast<uint32_t>(numBlocks);
        desc.numBlocks = static_cast<uint32_t>(numBlocks);
        desc.deviceBase = devicePtr + static_cast<uint64_t>(i) * static_cast<uint64_t>(cellsPerBuffer) * bufferBytes;
        desc.leaseStride = static_cast<uint32_t>(bufferBytes);
        desc.cellBytes = static_cast<uint32_t>(bufferBytes);
        desc.layout.numChunks = numChunks;
        desc.layout.chunkBytes = chunkBytes;
        desc.layout.chunkStride = chunkStride;
        desc.layout.unitStride = unitStride;
        desc.width = sparseCopyWidth(desc.layout, desc.hostPageStride, bufferBytes);
    }
    CUdeviceptr descPtr = 0;
    cuCheck(cuMemAlloc(&descPtr, sizeof(SparseCopyDesc) * descs.size()));
    CudaUniqPtr<> descOwner{reinterpret_cast<std::byte*>(descPtr)};
    cuCheck(cuMemcpyHtoD(descPtr, descs.data(), sizeof(SparseCopyDesc) * descs.size()));

    SparseUnitCopyArgs args;
    args.views = reinterpret_cast<SparseCacheView const*>(viewPtr);
    args.numRows = 1;
    args.copyDesc = reinterpret_cast<SparseCopyDesc const*>(descPtr);
    args.bufferBase = 0;
    args.numBuffers = static_cast<int32_t>(descs.size());
    args.direction = writeBack ? SparseCopyDirection::kWriteBack : SparseCopyDirection::kFetch;
    args.unitsPerPage = unitsPerPage;
    args.work.row = reinterpret_cast<int32_t*>(workRowBuf.get());
    args.work.ordinal = reinterpret_cast<int32_t*>(missOrd.get());
    args.work.slot = reinterpret_cast<int32_t*>(missSlotBuf.get());
    args.work.total = reinterpret_cast<int32_t*>(totalBuf.get());
    args.work.capacity = static_cast<int32_t>(flatOrdinal.size());
    args.work.numLists = numLists;
    args.bufferLayer = reinterpret_cast<int32_t*>(bufferLayerBuf.get());
    args.gridCtas = gridCtas;

    launchSparseUnitCopy(args, nullptr);
    cuCheck(cuStreamSynchronize(nullptr));

    // Write-back moves into the cold pool, so that is the side worth returning.
    CUdeviceptr const result = writeBack ? hostPtr : devicePtr;
    size_t const resultBytes = writeBack ? hostPool.size() : deviceArena.size();
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
    auto outRow = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outOrdinal = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outSlot = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outTotal = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto outCount = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));

    // The scan spans the batch, so the hook stands one row in for a single request.
    SparseCacheView view;
    view.residentOrdinal = reinterpret_cast<int32_t*>(resident.get());
    view.numLayers = numLayers;
    view.entriesPerLayer = entriesPerLayer;
    CUdeviceptr viewPtr = 0;
    cuCheck(cuMemAlloc(&viewPtr, sizeof(SparseCacheView)));
    CudaUniqPtr<> viewOwner{reinterpret_cast<std::byte*>(viewPtr)};
    cuCheck(cuMemcpyHtoD(viewPtr, &view, sizeof(SparseCacheView)));

    SparseFlushListArgs args;
    args.views = reinterpret_cast<SparseCacheView const*>(viewPtr);
    args.numRows = 1;
    args.numLayers = numLayers;
    args.fromUnits = reinterpret_cast<int32_t const*>(from.get());
    args.toUnits = reinterpret_cast<int32_t const*>(to.get());
    args.work.row = reinterpret_cast<int32_t*>(outRow.get());
    args.work.ordinal = reinterpret_cast<int32_t*>(outOrdinal.get());
    args.work.slot = reinterpret_cast<int32_t*>(outSlot.get());
    args.work.total = reinterpret_cast<int32_t*>(outTotal.get());
    args.work.capacity = maxFlushUnits;
    args.work.numLists = numLayers;
    args.outCount = reinterpret_cast<int32_t*>(outCount.get());

    launchSparseFlushList(args, nullptr);
    cuCheck(cuStreamSynchronize(nullptr));

    SparseFlushList result;
    result.row = readBack(outRow, listCount);
    result.ordinal = readBack(outOrdinal, listCount);
    result.slot = readBack(outSlot, listCount);
    result.total = readBack(outTotal, static_cast<size_t>(numLayers));
    result.count = readBack(outCount, static_cast<size_t>(numLayers));
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
