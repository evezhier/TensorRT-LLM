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

#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/storage/core.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

static double nowSeconds()
{
    return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
                                   .count())
        / 1e6;
}

// ---------------------------------------------------------------------------
// PageIndexConverter
// ---------------------------------------------------------------------------

std::vector<int> PageIndexConverter::operator()(
    std::vector<int> const& baseIndices, std::optional<PageIndexMode> indexMode, ScratchDesc const* scratch) const
{
    if (!indexMode.has_value())
    {
        if (scratch)
        {
            throw std::invalid_argument("index_mode must be provided when scratch is active");
        }
        indexMode = PageIndexMode::SHARED;
    }

    int appliedLayerOffset = (*indexMode == PageIndexMode::PER_LAYER) ? layerOffset : 0;
    int scratchPages = scratchPagesPerBlock;

    std::vector<int> result;
    result.reserve(baseIndices.size() * static_cast<size_t>(expansion));

    for (BlockOrdinal ordinal{0}; ordinal < BlockOrdinal{static_cast<int>(baseIndices.size())}; ++ordinal)
    {
        int index;
        if (scratch && scratch->range.contains(ordinal))
        {
            // Scratch block: slot IDs come from ScratchDesc, not base_indices.
            int blockPos = ordinal - scratch->range.beg;
            int totalOffset = blockPos * scratchPages;
            int slotIdx = totalOffset / scale;
            int slotId = scratch->slotIds[static_cast<size_t>(slotIdx)];
            int offset = totalOffset % scale;
            index = slotId * scale + (offset + appliedLayerOffset) % scale;
        }
        else if (baseIndices[toSizeT(ordinal)] == kBadPageIndex.value())
        {
            index = kBadPageIndex.value();
        }
        else
        {
            index = baseIndices[toSizeT(ordinal)] * scale + appliedLayerOffset;
        }
        for (int i = 0; i < expansion; ++i)
        {
            result.push_back(index != kBadPageIndex.value() ? index * expansion + i : kBadPageIndex.value());
        }
    }
    return result;
}

std::vector<int> PageIndexConverter::operator()(int baseIndex) const
{
    return operator()(std::vector<int>{baseIndex}, std::nullopt, nullptr);
}

// ---------------------------------------------------------------------------
// KvCacheManager
// ---------------------------------------------------------------------------

uint32_t KvCacheManager::numLiveManagers() noexcept
{
    return PoisonHold::count();
}

KvCacheManager::KvCacheManager(KVCacheManagerConfig const& config, std::shared_ptr<EventSink> eventSink,
    std::unique_ptr<IKvCacheColdPageCodec> coldPageCodec)
    : mConfig(config)
    , mLifeCycles(config)
    , mEventSink(std::move(eventSink))
    , mAvgReusedLength(0.9999)
    , mAvgSqrCapacity(0.9999)
    , mAvgSqrHistoryLength(0.9999)
{
    mConfig.validate();

    mRadixTree = std::make_shared<BlockRadixTree>(mLifeCycles, mConfig.tokensPerBlock, mEventSink);

    StorageConfig storageConfig = createStorageConfig(mConfig);
    mStorage = std::make_shared<StorageManager>(mLifeCycles, storageConfig, mConfig.tokensPerBlock,
        std::move(coldPageCodec), mConfig.swaScratchReuse, mConfig.typicalStep, mConfig.constraints,
        mConfig.initialPoolRatio, mEventSink, mConfig.maxUtilForResume);

    _computeSparseCacheShape();
    _buildSparseArenas();

    mTargetRatioListHot = _currentHotRatio();
    mTargetRatioListCold = _currentColdRatios();
    _resetIterationPeakNumBlocks();

    mLastAdjustmentTime = nowSeconds();
}

KvCacheManager::~KvCacheManager()
{
    KVCM2_POISON_ON_EXCEPT([this]() { shutdown(); });
}

void KvCacheManager::_computeSparseCacheShape()
{
    LifeCycleId const numLc = mStorage->numLifeCycles();
    mSparseCacheMaxUnits = TypedVec<LifeCycleId, int>(numLc, 0);
    mSparseCacheExpansion = TypedVec<LifeCycleId, int>(numLc, 1);
    mSparseCacheBuffers = TypedVec<LifeCycleId, int>(numLc, 0);
    mSparseCacheSelection = TypedVec<LifeCycleId, int>(numLc, 0);
    mSparseCacheLayers = TypedVec<LifeCycleId, int>(numLc, 0);

    for (auto const& layer : mConfig.layers)
    {
        auto const* attn = std::get_if<AttentionLayerConfig>(&layer);
        if (attn == nullptr)
        {
            continue;
        }
        LifeCycleId const lcId = mLifeCycles.getId(makeLifeCycle(layer, mConfig.tokensPerBlock));
        bool layerCounted = false;
        for (auto const& buf : attn->buffers)
        {
            if (!buf.isSparse || buf.maxCachedUnits <= 0)
            {
                continue;
            }
            int const expansion = mConfig.tokensPerBlock / buf.tokensPerBlockOverride.value_or(mConfig.tokensPerBlock);
            mSparseCacheMaxUnits[lcId] = std::max(mSparseCacheMaxUnits[lcId], buf.maxCachedUnits);
            mSparseCacheSelection[lcId] = std::max(mSparseCacheSelection[lcId], buf.numSelectedBlocks);
            mSparseCacheExpansion[lcId] = std::max(mSparseCacheExpansion[lcId], expansion);
            // One more cell in a reserved slot.
            ++mSparseCacheBuffers[lcId];
            if (!layerCounted)
            {
                // One more metadata row: the layer's buffers share it.
                ++mSparseCacheLayers[lcId];
                layerCounted = true;
            }
        }
    }
}

int KvCacheManager::sparseCacheSubSlotsPerBuffer(LifeCycleId lifeCycle) const noexcept
{
    int const maxUnits = sparseCacheMaxUnits(lifeCycle);
    if (maxUnits <= 0)
    {
        return 0;
    }
    // maxCachedUnits counts both populations, the history this caches and everything still
    // in flight, so the reservation follows from configuration alone.
    return divUp(maxUnits, sparseCacheExpansion(lifeCycle));
}

int KvCacheManager::sparseCacheMaxInFlightUnits(LifeCycleId lifeCycle) const noexcept
{
    int const cells = sparseCacheSubSlotsPerBuffer(lifeCycle) * sparseCacheExpansion(lifeCycle);
    if (cells <= 0)
    {
        return 0;
    }
    // A step advances capacity one token past history, and one token straddles at most one
    // unit boundary, so at most one unit is ever in flight. That is the whole of the
    // headroom: everything else the reservation holds is available to the selection.
    return 1;
}

std::vector<SparseBufferPlan> KvCacheManager::sparseBufferPlan(LifeCycleId lifeCycle) const
{
    // Layers outermost, as _computeSparseCacheShape walks them, so a layer's buffers stay
    // consecutive -- the fetch resolves one layer and names its buffers as a range.
    std::vector<SparseBufferPlan> plan;
    std::optional<LayerId> previousLayer;
    int32_t layerRow = -1;
    for (auto const& layer : mConfig.layers)
    {
        auto const* attn = std::get_if<AttentionLayerConfig>(&layer);
        if (attn == nullptr || mLifeCycles.getId(makeLifeCycle(layer, mConfig.tokensPerBlock)) != lifeCycle)
        {
            continue;
        }
        auto const firstOfLayer = plan.size();
        for (auto const& buf : attn->buffers)
        {
            if (!buf.isSparse || buf.maxCachedUnits <= 0)
            {
                continue;
            }
            if (!previousLayer.has_value() || attn->layerId != *previousLayer)
            {
                previousLayer = attn->layerId;
                ++layerRow;
            }
            // Guaranteed by BufferConfig::validate: a cached sparse buffer carries a layout.
            TLLM_CHECK(buf.sparseUnitLayout.has_value());
            auto const attr = bufferAttr(attn->layerId, buf.role);
            plan.push_back(SparseBufferPlan{BufferId{attn->layerId, buf.role}, attr.poolIndex, attr.size,
                static_cast<uint32_t>(attr.offset), *buf.sparseUnitLayout, layerRow, 0});
        }
        // Group a layer's buffers by pool so a (layer, pool) run is consecutive too: a pool is
        // what fixes a cell's size, the slot stride it is read from, and the unit layout, so
        // the copy addresses one pool per launch and needs to name its buffers as a range.
        // Stable, so declaration order survives within a pool.
        std::stable_sort(plan.begin() + static_cast<ptrdiff_t>(firstOfLayer), plan.end(),
            [](SparseBufferPlan const& a, SparseBufferPlan const& b) { return a.poolIndex < b.poolIndex; });
    }

    // Where each buffer's cells sit inside its pool's arena region.
    std::map<PoolIndex, int32_t> nextSlot;
    for (auto& entry : plan)
    {
        entry.slotInPool = nextSlot[entry.poolIndex]++;
    }
    return plan;
}

std::vector<BufferId> KvCacheManager::sparseBufferIds(LifeCycleId lifeCycle) const
{
    std::vector<BufferId> ids;
    for (auto const& entry : sparseBufferPlan(lifeCycle))
    {
        ids.push_back(entry.id);
    }
    return ids;
}

void KvCacheManager::_buildSparseArenas()
{
    LifeCycleId const numLc = mStorage->numLifeCycles();
    mSparseArenas = TypedVec<LifeCycleId, SparseCellArena>(numLc);

    // Concurrent requests the arena must serve. `constraints` is the existing declaration of
    // the batch shape, and the one Batch already derives its rows from, so a sparse
    // deployment sizes both from one place.
    int maxLeases = 0;
    for (auto const& desc : mConfig.constraints)
    {
        maxLeases = std::max(maxLeases, static_cast<int>(desc.kvCaches.size()));
    }

    for (LifeCycleId lcId{0}; lcId < numLc; ++lcId)
    {
        if (sparseCacheMaxUnits(lcId) <= 0)
        {
            continue;
        }
        if (maxLeases <= 0)
        {
            throw std::invalid_argument(
                "A sparse buffer needs `constraints` to declare how many requests run together; "
                "the cache arena is reserved up front and cannot be sized without it");
        }
        auto const plan = sparseBufferPlan(lcId);
        TLLM_CHECK(!plan.empty());

        SparseCacheInit shape;
        shape.numLayers = sparseCacheLayerCount(lcId);
        shape.numBuffers = sparseCacheBufferCount(lcId);
        shape.cellsPerBuffer = sparseCacheSubSlotsPerBuffer(lcId);
        shape.entriesPerLayer = shape.cellsPerBuffer * sparseCacheExpansion(lcId);
        shape.maxFlushUnits = sparseCacheMaxInFlightUnits(lcId);
        shape.maxFlushBlocks = divUp(shape.maxFlushUnits, sparseCacheExpansion(lcId)) + 1;

        // The arena mirrors the hot pool group: one region per pool the life cycle's sparse
        // buffers draw on, sized by that pool's cell and the buffers sharing it.
        size_t numPools = 0;
        for (auto const& entry : plan)
        {
            numPools = std::max(numPools, static_cast<size_t>(entry.poolIndex.value()) + 1);
        }
        shape.pools.assign(numPools, {0, 0});
        for (auto const& entry : plan)
        {
            shape.pageBufferOffset.push_back(entry.pageOffset);
            shape.bufferLayer.push_back(entry.layerRow);
            shape.bufferSlotInPool.push_back(entry.slotInPool);
            auto& [cellBytes, buffers] = shape.pools[static_cast<size_t>(entry.poolIndex.value())];
            cellBytes = entry.size;
            ++buffers;
        }
        // A pool index the life cycle never uses still gets a region, because the arena is
        // indexed by the storage pool index rather than by a dense numbering of its own.
        for (auto& [cellBytes, buffers] : shape.pools)
        {
            if (buffers == 0)
            {
                cellBytes = 1;
                buffers = 1;
            }
        }
        mSparseArenas[lcId] = SparseCellArena(shape, maxLeases);
    }
}

std::vector<SparseUnitCopyArgs> KvCacheManager::sparseCopyTemplates(LifeCycleId lifeCycle) const
{
    auto const plan = sparseBufferPlan(lifeCycle);
    TLLM_CHECK(!plan.empty());
    PoolGroupIndex const hotPg = mStorage->getPoolGroupIndex(kHotLevel, lifeCycle);
    auto const& arena = mSparseArenas[lifeCycle];

    std::vector<SparseUnitCopyArgs> templates;
    for (size_t begin = 0; begin < plan.size();)
    {
        auto const pool = plan[begin].poolIndex;
        size_t end = begin;
        while (end < plan.size() && plan[end].poolIndex == pool)
        {
            ++end;
        }

        // A pool holds buffers of one size, but not necessarily of one unit layout, and the
        // copy walks a run with one set of strides.
        auto const& layout = plan[begin].layout;
        for (size_t i = begin; i < end; ++i)
        {
            auto const& other = plan[i].layout;
            if (other.numChunks != layout.numChunks || other.chunkBytes != layout.chunkBytes
                || other.chunkStride != layout.chunkStride || other.unitStride != layout.unitStride)
            {
                throw LogicError("Sparse buffers sharing a pool must share a selection unit layout");
            }
        }

        SparseUnitCopyArgs args;
        // The page side is the life cycle's own hot pool, in ordinary slot layout. It is host
        // memory, registered device-mapped, so the kernel addresses it directly; but its layout
        // is KVCM2's rather than a codec's, which is what keeps the fetch independent of
        // whatever a codec does on the way to level 1.
        args.hostBase = static_cast<CUdeviceptr>(mStorage->getMemPoolBaseAddress(hotPg, pool));
        args.hostPageStride = mStorage->slotSize(hotPg).at(pool);
        // The hot side is the cache arena, not a storage pool: a cell is the only GPU-resident
        // copy a kernel can read, and the arena is where cells live.
        args.deviceBase = arena.base(pool);
        args.bufferBytes = arena.bufferBytes(pool);
        args.cellsPerBuffer = sparseCacheSubSlotsPerBuffer(lifeCycle);
        args.cellsPerLease = arena.cellsPerLease(pool);
        args.expansion = sparseCacheExpansion(lifeCycle);
        args.layout.numChunks = layout.numChunks;
        args.layout.chunkBytes = layout.chunkBytes;
        args.layout.chunkStride = layout.chunkStride;
        args.layout.unitStride = layout.unitStride;
        args.bufferBase = static_cast<int32_t>(begin);
        args.numBuffers = static_cast<int32_t>(end - begin);
        templates.push_back(args);
        begin = end;
    }
    return templates;
}

MemAddress KvCacheManager::getSparseCacheBaseAddress(LayerId layerId, DataRole const& role) const
{
    LifeCycleId const lcId{getLayerGroupId(layerId).value()};
    auto const& arena = mSparseArenas[lcId];
    auto const plan = sparseBufferPlan(lcId);
    auto const it = std::find_if(plan.begin(), plan.end(),
        [&](SparseBufferPlan const& entry) { return entry.id.layerId == layerId && entry.id.role == role; });
    if (arena.empty() || it == plan.end())
    {
        throw std::invalid_argument("getSparseCacheBaseAddress: this buffer has no sparse fetch path");
    }
    // Per pool, not per buffer: the arena mirrors the hot pool group, and an index the fetch
    // emits already carries which buffer of the pool it belongs to.
    return static_cast<MemAddress>(arena.base(it->poolIndex));
}

void KvCacheManager::fetchSparsePages(Batch& batch, std::vector<BufferId> const& buffers, int32_t const* selected,
    int selectionWidth, int32_t* out, CUstream stream)
{
    if (buffers.empty())
    {
        throw std::invalid_argument("fetchSparsePages: no buffers");
    }
    if (selected == nullptr || out == nullptr)
    {
        throw std::invalid_argument("fetchSparsePages: selected and out must be device pointers");
    }

    LayerId const layerId = buffers.front().layerId;
    for (auto const& id : buffers)
    {
        if (id.layerId != layerId)
        {
            throw std::invalid_argument("fetchSparsePages: every buffer must belong to one layer");
        }
    }

    LifeCycleId const lcId{getLayerGroupId(layerId).value()};
    auto const all = sparseBufferIds(lcId);
    // Buffers are grouped by layer, so this layer's are a contiguous run.
    auto const first
        = std::find_if(all.begin(), all.end(), [layerId](BufferId const& id) { return id.layerId == layerId; });
    auto const last = std::find_if(first, all.end(), [layerId](BufferId const& id) { return id.layerId != layerId; });
    if (!std::equal(first, last, buffers.begin(), buffers.end()))
    {
        throw std::invalid_argument(
            "fetchSparsePages: buffers must be exactly this layer's sparse buffers, in the cache's order");
    }

    auto const bufferBase = static_cast<int32_t>(std::distance(all.begin(), first));
    auto const numBuffers = static_cast<int32_t>(std::distance(first, last));
    // The metadata row is the layer's index among those carrying a cache, which is the
    // number of distinct layers before it in the same walk.
    int32_t layerInGroup = 0;
    for (auto it = all.begin(); it != first; ++it)
    {
        if (it == all.begin() || it->layerId != std::prev(it)->layerId)
        {
            ++layerInGroup;
        }
    }

    auto const& scratch = batch.sparseFetchScratch(LayerGroupId{lcId.value()});
    if (scratch.outSlot == nullptr)
    {
        throw std::invalid_argument("fetchSparsePages: this layer group carries no sparse cache");
    }
    if (selectionWidth != scratch.selectionWidth)
    {
        throw std::invalid_argument("fetchSparsePages: selectionWidth must match the configured numSelectedBlocks");
    }

    int const expansion = sparseCacheExpansion(lcId);
    LayerGroupId const lgId{lcId.value()};
    auto const views = reinterpret_cast<SparseCacheView const*>(batch.sparseCacheViews(lgId));
    auto const watermarks = batch.watermarks();
    auto const table = batch.pageTable(lgId);
    auto const numRows = batch.maxBatchSize();

    SparseResolveArgs resolve;
    resolve.views = views;
    resolve.numRows = numRows;
    resolve.layerInGroup = layerInGroup;
    resolve.numLayers = sparseCacheLayerCount(lcId);
    resolve.entriesPerLayer = sparseCacheSubSlotsPerBuffer(lcId) * expansion;
    resolve.selected = selected;
    resolve.selectionWidth = selectionWidth;
    resolve.historyTokens = reinterpret_cast<int32_t const*>(watermarks.historyLength);
    resolve.capacityTokens = reinterpret_cast<int32_t const*>(watermarks.capacity);
    resolve.unitTokens = mConfig.tokensPerBlock / expansion;
    resolve.outSlot = scratch.outSlot;
    resolve.inFlightSlot = scratch.inFlightSlot;
    resolve.maxInFlightUnits = scratch.maxInFlightUnits;
    resolve.missOrdinal = scratch.missOrdinal;
    resolve.missSlot = scratch.missSlot;
    resolve.missCount = scratch.missCount;
    launchSparseResolve(resolve, stream);

    // The misses this layer produced, applied to each of its buffers. One list row per
    // batch row, so the inner index is zero once workRowBase names the layer resolved.
    //
    // One pass per pool: a pool fixes the cell size and the slot stride, so each is its own
    // launch. The templates span the life cycle, so each is clipped to this layer's buffers.
    for (auto copy : sparseCopyTemplates(lcId))
    {
        auto const runBase = std::max(copy.bufferBase, bufferBase);
        auto const runEnd = std::min(copy.bufferBase + copy.numBuffers, bufferBase + numBuffers);
        if (runBase >= runEnd)
        {
            continue;
        }
        copy.bufferBase = runBase;
        copy.numBuffers = runEnd - runBase;
        copy.views = views;
        copy.numRows = numRows;
        copy.pageTable = reinterpret_cast<int32_t const*>(table.data);
        copy.pageTableStride = table.blocksPerRow;
        copy.numBlocks = table.blocksPerRow;
        copy.pageTableBase = 0;
        copy.direction = SparseCopyDirection::kFetch;
        copy.workOrdinal = scratch.missOrdinal;
        copy.workSlot = scratch.missSlot;
        copy.workCount = scratch.missCount;
        copy.workWidth = selectionWidth;
        copy.workRowStride = 1;
        copy.workRowBase = layerInGroup;
        launchSparseUnitCopy(copy, stream);

        SparseOutIndexArgs indices;
        indices.views = views;
        indices.numRows = numRows;
        indices.selected = selected;
        indices.outSlot = scratch.outSlot;
        indices.selectionWidth = selectionWidth;
        indices.out = out;
        indices.outBuffers = numBuffers;
        indices.outBase = runBase - bufferBase;
        indices.bufferBase = copy.bufferBase;
        indices.numBuffers = copy.numBuffers;
        indices.cellsPerBuffer = copy.cellsPerBuffer;
        indices.cellsPerLease = copy.cellsPerLease;
        indices.expansion = expansion;
        launchSparseOutIndices(indices, stream);
    }
}

void KvCacheManager::_checkNoLivingKvCaches(char const* api) const
{
    TLLM_CHECK_WITH_INFO(mLivingKvCaches.empty(),
        "%s with %zu KV cache(s) still open; close them (or drain the engine) first", api, mLivingKvCaches.size());
}

void KvCacheManager::shutdown()
{
    // Disposal, not work: a caller that detects corruption still has to be able to tear the
    // manager down, so this succeeds while poisoned instead of rejecting. The cleanup below is
    // skipped for the same reason every poisoned destructor skips its own -- the structures it
    // would walk are known to be inconsistent -- so the manager's memory is leaked deliberately.
    //
    // The latch is read under the lock because it is normally set by a thread holding it: a read
    // taken before blocking would only say the cache was intact before this call started waiting.
    auto const apiLock = lockExclusive();
    if (Poison::poisoned())
    {
        return;
    }
    _checkNoLivingKvCaches("shutdown()");
    clearReusableBlocks();
    TLLM_CHECK_DEBUG(mStorage);

    // Post-condition: after clearing all reusable blocks and with no active KvCaches,
    // no evictable pages should remain.
    for (auto const& lvl : mStorage->mLevels)
    {
        for (PoolGroupIndex pgIdx{0}; pgIdx < lvl.numPoolGroups(); ++pgIdx)
        {
            TLLM_CHECK_DEBUG(lvl.controller.numEvictablePages(pgIdx) == 0);
        }
    }

    mStorage->destroy();
}

void KvCacheManager::clearReusableBlocks()
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    _checkNoLivingKvCaches("clear_reusable_blocks()");
    TLLM_CHECK_DEBUG(mRadixTree);
    mRadixTree->clear();
}

std::shared_ptr<KvCache> KvCacheManager::createKvCache(ReuseScope reuseScope, TokenSpan inputTokens,
    std::optional<RequestIdType> id, KvCache::PriorityCb priorityCb, std::optional<int> expectedPromptLength,
    std::optional<bool> textOnly, bool enableRequestStats)
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    if (!priorityCb)
    {
        priorityCb = [](BlockOrdinal, LifeCycleId) { return kPriorityDefault; };
    }

    if (!expectedPromptLength.has_value() && inputTokens.size() != 0)
    {
        expectedPromptLength = static_cast<int>(inputTokens.size());
    }

    // Compute the reuse match once here (shared conceptually with probeReuse) and
    // hand it to the KvCache, rather than having the cache re-walk the radix tree.
    // Mirrors Python KVCacheManager.allocate() passing a ReuseMatch into _KVCache.
    std::optional<BlockRadixTree::ReuseMatch> reuseMatch;
    if (inputTokens.size() != 0)
    {
        reuseMatch = matchReuse(reuseScope, inputTokens, textOnly.value_or(this->textOnly()));
    }

    return std::make_shared<KvCache>(*this, std::move(reuseScope), std::move(reuseMatch), std::move(id),
        std::move(priorityCb), expectedPromptLength, textOnly, enableRequestStats);
}

BlockRadixTree::ReuseMatch KvCacheManager::matchReuse(
    ReuseScope const& reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    KVCM2_REJECT_IF_POISONED();
    return mRadixTree->match(reuseScope, inputTokens, knownNoDigest, enablePartialMatch(), mConfig.reuseMatchBackoff);
}

int KvCacheManager::probeReuse(ReuseScope reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    KVCM2_REJECT_IF_POISONED();
    // Shared: the traversal is read-only and the matched Block* never escape this call.
    auto const apiLock = lockShared();
    return matchReuse(reuseScope, inputTokens, knownNoDigest).numTokens;
}

std::optional<BlockKey> KvCacheManager::probeFirstNewBlockKey(
    ReuseScope reuseScope, TokenSpan inputTokens, bool knownNoDigest) const
{
    // The matched Block* remain protected until the key has been copied out.
    // Call matchReuse directly: probeReuse would nest shared-lock acquisitions.
    auto const apiLock = lockShared();
    auto const match = matchReuse(reuseScope, inputTokens, knownNoDigest);
    int const blockSize = tokensPerBlock();
    int const blockIndex = match.numTokens / blockSize;
    size_t const begin = static_cast<size_t>(blockIndex) * static_cast<size_t>(blockSize);
    if (begin + static_cast<size_t>(blockSize) > static_cast<size_t>(inputTokens.size()))
    {
        return std::nullopt;
    }

    // Derive the boundary from the final, pruned match, not the raw token path.
    // A partial match can end in a block with a different suffix, so only the
    // preceding fully matched block supplies the query's exact prefix key.
    BlockKey const previousKey
        = blockIndex == 0 ? RootBlock::makeKey(reuseScope) : match.blocks[BlockOrdinal{blockIndex - 1}]->key;
    return Block::makeKey(previousKey, inputTokens.begin() + begin, static_cast<size_t>(blockSize), knownNoDigest);
}

// ---- Memory pool queries --------------------------------------------------

MemAddress KvCacheManager::getMemPoolBaseAddress(
    LayerId layerId, DataRole role, std::optional<PageIndexMode> indexMode) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);

    if (!indexMode.has_value())
    {
        if (mConfig.enableSwaScratchReuse())
        {
            throw std::invalid_argument("index_mode must be provided when SWA scratch reuse is enabled");
        }
        indexMode = PageIndexMode::SHARED;
    }

    PoolGroupIndex pgIdx = mStorage->getPoolGroupIndex(attr.lifeCycleId);
    MemAddress addr = mStorage->getMemPoolBaseAddress(pgIdx, attr.poolIndex);
    if (*indexMode == PageIndexMode::SHARED)
    {
        addr = MemAddress(addr + attr.offset);
    }
    return addr;
}

int KvCacheManager::getPageStride(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return exactDiv(static_cast<int>(attr.size), attr.expansion);
}

size_t KvCacheManager::getPageIndexUpperBound(LayerId layerId, DataRole role) const
{
    auto const apiLock = lockShared();
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    LifeCycleId lc = attr.lifeCycleId;
    PoolGroupIndex pg = mStorage->getPoolGroupIndex(lc);
    SlotCount const numSlots = mStorage->numSlots(pg, kHotLevel);
    auto slotSizes = mStorage->slotSize(pg);
    size_t slotSize = slotSizes.at(attr.poolIndex);
    return (exactDiv(slotSize, attr.size) * slotCountToSizeT(numSlots) - exactDiv(attr.offset, attr.size))
        * static_cast<size_t>(attr.expansion);
}

int KvCacheManager::getPageIndexScale(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    return mStorage->mSlotToPageIndices.at(attr.lifeCycleId).at(attr.poolIndex);
}

PageIndexConverter KvCacheManager::getPageIndexConverter(LayerId layerId, DataRole role) const
{
    auto const& attr = mStorage->getBufferAttr(layerId, role);
    auto const& layerAttr = mStorage->getLayerAttr(layerId);
    int scale = mStorage->mSlotToPageIndices.at(attr.lifeCycleId).at(attr.poolIndex);
    int offset = exactDiv(static_cast<int>(attr.offset), static_cast<int>(attr.size));
    int scratchPages = layerAttr.slotUtil.at(attr.poolIndex);
    return PageIndexConverter{scale, attr.expansion, offset, scratchPages};
}

std::optional<bool> KvCacheManager::supportsIndexMode(PageIndexMode mode) const
{
    switch (mode)
    {
    case PageIndexMode::PER_LAYER: return true;
    case PageIndexMode::SHARED: return mConfig.enableSwaScratchReuse() ? std::optional<bool>(std::nullopt) : true;
    }
    return std::nullopt;
}

// ---- getAggregatedPages ---------------------------------------------------

std::vector<AggregatedPageDesc> KvCacheManager::getAggregatedPages(std::vector<BufferId> const& buffers) const
{
    auto const apiLock = lockShared();
    using Key = std::pair<LifeCycleId, PoolIndex>;

    struct Entry
    {
        size_t start;
        size_t end;
        ExpandedBuffer eb;
    };

    std::map<Key, std::vector<Entry>> groups;

    for (auto const& bufferId : buffers)
    {
        auto it = mStorage->mBufferAttr.find(bufferId);
        if (it == mStorage->mBufferAttr.end())
            throw std::out_of_range("getAggregatedPages: unknown buffer id");

        auto const& attr = it->second;
        size_t start = attr.offset;
        size_t end = attr.offset + attr.size;
        Key key{attr.lifeCycleId, attr.poolIndex};
        groups[key].push_back({start, end, ExpandedBuffer{bufferId, attr.expansion}});
    }

    std::vector<AggregatedPageDesc> result;
    for (auto& [key, entries] : groups)
    {
        auto [lifeCycleId, poolIdx] = key;
        auto pgIdx = mStorage->getPoolGroupIndex(lifeCycleId);

        std::sort(entries.begin(), entries.end(), [](Entry const& a, Entry const& b) { return a.start < b.start; });

        auto const poolBase = mStorage->getMemPoolBaseAddress(pgIdx, poolIdx);
        size_t stride = mStorage->slotSize(pgIdx).at(poolIdx);

        auto flush
            = [&, lifeCycleId = lifeCycleId](size_t start, size_t end, std::vector<ExpandedBuffer>& buffersInRange)
        {
            result.push_back(AggregatedPageDesc{
                MemAddress(poolBase + start), end - start, stride, lifeCycleId, std::move(buffersInRange)});
        };

        size_t currentStart = entries.front().start;
        size_t currentEnd = entries.front().end;
        std::vector<ExpandedBuffer> currentBuffers{entries.front().eb};
        for (size_t i = 1; i < entries.size(); ++i)
        {
            if (entries[i].start == currentEnd)
            {
                currentEnd = entries[i].end;
                currentBuffers.push_back(entries[i].eb);
                continue;
            }

            flush(currentStart, currentEnd, currentBuffers);
            currentStart = entries[i].start;
            currentEnd = entries[i].end;
            currentBuffers = {entries[i].eb};
        }
        flush(currentStart, currentEnd, currentBuffers);
    }

    return result;
}

// ---- Pool group layout -----------------------------------------------------

TypedVec<PoolGroupIndex, PoolGroupDesc> KvCacheManager::poolGroupDescs() const
{
    // Shared: PoolGroupDesc carries numSlots, which _adjustLevel changes on a pool resize.
    auto const apiLock = lockShared();
    return mStorage->poolGroupDescs();
}

// ---- Query / info ---------------------------------------------------------

int KvCacheManager::tokensPerBlock() const noexcept
{
    return mRadixTree->tokensPerBlock();
}

bool KvCacheManager::enablePartialMatch() const noexcept
{
    return mConfig.enablePartialReuse;
}

bool KvCacheManager::enablePartialCommit() const noexcept
{
    return mConfig.enablePartialCommit;
}

int KvCacheManager::numLayers() const noexcept
{
    return static_cast<int>(mStorage->layerToLifeCycleIds().size());
}

std::vector<LayerId> KvCacheManager::layerIds() const
{
    std::vector<LayerId> ids;
    for (auto const& [lid, lc] : mStorage->layerToLifeCycleIds())
        ids.push_back(lid);
    return ids;
}

LayerGroupId KvCacheManager::getLayerGroupId(LayerId layerId) const
{
    return mStorage->layerToLifeCycleIds().at(layerId);
}

TypedVec<LayerGroupId, std::vector<LayerId>> KvCacheManager::layerGrouping() const
{
    LifeCycleId numLc = mLifeCycles.size();
    TypedVec<LayerGroupId, std::vector<LayerId>> result(numLc);
    for (auto const& [lid, lc] : mStorage->layerToLifeCycleIds())
    {
        result.at(lc).push_back(lid);
    }
    return result;
}

// ---- Resize ---------------------------------------------------------------

bool KvCacheManager::resize(CacheLevel level, size_t quota, bool bestEfforts)
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    // Same precondition as adjust(): _adjustLevel may defragment, invalidating any page index an
    // ACTIVE cache holds.
    for (KvCache* kvc : mLivingKvCaches)
        TLLM_CHECK_WITH_INFO(kvc->status() == KvCache::Status::SUSPENDED,
            "level adjustment requires every KvCache to be SUSPENDED: _adjustLevel may "
            "defragment, which relocates pages and invalidates indices an ACTIVE cache holds");
    if (bestEfforts)
        throw std::runtime_error("best_efforts resize not implemented");
    try
    {
        _adjustLevel(level, quota);
        return true;
    }
    catch (std::exception const& e)
    {
        TLLM_LOG_WARNING("Failed to resize cache level %d to %zu: %s", level.value(), quota, e.what());
        return false;
    }
}

size_t KvCacheManager::getQuota(CacheLevel level) const
{
    auto const apiLock = lockShared();
    return mStorage->mLevels.at(level).storage->totalQuota();
}

// ---- Statistics ----------------------------------------------------------

void KvCacheManager::commitStats(
    KVCacheStatsDelta const& stats, IterationStatsByLifeCycle const& iterationStatsByLifeCycle)
{
    if (!mConfig.enableStats)
    {
        return;
    }

    _updateIterationPeakNumBlocks();
    mCommittedStats.add(stats);
    for (auto const& [lifeCycle, iterationStats] : iterationStatsByLifeCycle)
    {
        if (!iterationStats.empty())
        {
            mIterationStatsByLifeCycle[lifeCycle].add(iterationStats);
        }
    }
}

KVCacheStatsDelta KvCacheManager::getCommittedStats() const
{
    auto const apiLock = lockShared();
    return mCommittedStats.copy();
}

IterationStatsByLifeCycle KvCacheManager::getAndResetIterationStats()
{
    auto const apiLock = lockExclusive();
    IterationStatsByLifeCycle stats;
    for (auto const& [lifeCycle, delta] : mIterationStatsByLifeCycle)
    {
        if (!delta.empty())
        {
            stats.emplace(lifeCycle, delta.copy());
        }
    }
    mIterationStatsByLifeCycle.clear();
    return stats;
}

void KvCacheManager::commitSsmSnapshotIterationStats(SsmSnapshotIterationStatsByLifeCycle const& statsByLifeCycle)
{
    if (!mConfig.enableStats)
    {
        return;
    }
    for (auto const& [lifeCycle, iterationStats] : statsByLifeCycle)
    {
        if (!iterationStats.empty())
        {
            mSsmSnapshotIterationStatsByLifeCycle[lifeCycle].add(iterationStats);
        }
    }
}

SsmSnapshotIterationStatsByLifeCycle KvCacheManager::getAndResetSsmSnapshotIterationStats()
{
    auto const lock = lockExclusive();
    SsmSnapshotIterationStatsByLifeCycle stats;
    for (auto const& [lifeCycle, delta] : mSsmSnapshotIterationStatsByLifeCycle)
    {
        if (!delta.empty())
        {
            stats.emplace(lifeCycle, delta.copy());
        }
    }
    mSsmSnapshotIterationStatsByLifeCycle.clear();
    return stats;
}

TypedVec<PoolGroupIndex, StorageStatistics> KvCacheManager::getStorageStatistics(CacheLevel cacheLevel) const
{
    auto const apiLock = lockShared();
    TypedVec<PoolGroupIndex, StorageStatistics> result;
    PoolGroupIndex const numPoolGroups = mStorage->numPoolGroups(cacheLevel);
    result.reserve(numPoolGroups);
    for (PoolGroupIndex poolGroup{0}; poolGroup < numPoolGroups; ++poolGroup)
    {
        result.push_back(mStorage->getStatistics(cacheLevel, poolGroup));
    }
    return result;
}

TypedVec<LifeCycleId, PoolGroupIndex> KvCacheManager::getLifeCyclePoolGroupIndices(CacheLevel cacheLevel) const
{
    TypedVec<LifeCycleId, PoolGroupIndex> result;
    result.reserve(mLifeCycles.size());
    for (LifeCycleId lifeCycle{0}; lifeCycle < mLifeCycles.size(); ++lifeCycle)
    {
        result.push_back(mStorage->getPoolGroupIndex(cacheLevel, lifeCycle));
    }
    return result;
}

void KvCacheManager::commitReusedBlocksByLevel(ReusedBlocksByLevelByLifeCycle const& byLifeCycle)
{
    for (auto const& [lifeCycle, byLevel] : byLifeCycle)
    {
        if (!byLevel.empty())
        {
            mIterReusedBlocksByLevel[lifeCycle].add(byLevel);
        }
    }
}

ReusedBlocksByLevelByLifeCycle KvCacheManager::getAndResetIterationReusedBlocksByLevel()
{
    auto const apiLock = lockExclusive();
    ReusedBlocksByLevelByLifeCycle byLifeCycle;
    byLifeCycle.swap(mIterReusedBlocksByLevel);
    return byLifeCycle;
}

void KvCacheManager::recordRequestSuspended()
{
    auto const apiLock = lockExclusive();
    if (!mConfig.enableStats)
    {
        return;
    }
    ++mIterSuspendedRequests;
}

void KvCacheManager::recordRequestResumed()
{
    auto const apiLock = lockExclusive();
    if (!mConfig.enableStats)
    {
        return;
    }
    ++mIterResumedRequests;
}

void KvCacheManager::recordDiskPrefetchBlocks(int64_t numBlocks)
{
    TLLM_CHECK_DEBUG(numBlocks >= 0);
    if (!mConfig.enableStats)
    {
        return;
    }
    mIterDiskPrefetchBlocks += numBlocks;
}

int64_t KvCacheManager::getAndResetIterationDiskPrefetchBlocks()
{
    auto const apiLock = lockExclusive();
    auto const numBlocks = mIterDiskPrefetchBlocks;
    mIterDiskPrefetchBlocks = 0;
    return numBlocks;
}

void KvCacheManager::commitCachedTokensByLevel(CountsByLevel const& counts)
{
    TLLM_CHECK_DEBUG(std::all_of(counts.begin(), counts.end(), [](int64_t count) { return count >= 0; }));
    if (!mConfig.enableStats)
    {
        return;
    }
    addCountsByLevel(mIterCachedTokensByLevel, counts);
}

CountsByLevel KvCacheManager::getAndResetIterationCachedTokensByLevel()
{
    auto const apiLock = lockExclusive();
    CountsByLevel counts;
    std::swap(counts, mIterCachedTokensByLevel);
    return counts;
}

std::pair<int64_t, int64_t> KvCacheManager::getAndResetIterationSuspendResumeStats()
{
    auto const apiLock = lockExclusive();
    // Suspend/resume is a per-request, manager-level event (not per-pool-group), so it
    // is drained alongside getAndResetIterationStats once per iteration-stats fetch.
    auto const suspended = mIterSuspendedRequests;
    auto const resumed = mIterResumedRequests;
    mIterSuspendedRequests = 0;
    mIterResumedRequests = 0;
    return {suspended, resumed};
}

PeakBlockStatsByCacheLevel KvCacheManager::_currentBlockStatsByCacheLevel() const
{
    PeakBlockStatsByCacheLevel result(mStorage->numCacheLevels());
    for (CacheLevel cacheLevel{0}; cacheLevel < mStorage->numCacheLevels(); ++cacheLevel)
    {
        auto& levelStats = result[cacheLevel];
        levelStats.resize(mStorage->numPoolGroups(cacheLevel));
        for (PoolGroupIndex poolGroup{0}; poolGroup < mStorage->numPoolGroups(cacheLevel); ++poolGroup)
        {
            auto const stats = mStorage->getStatistics(cacheLevel, poolGroup);
            levelStats[poolGroup] = {stats.available(), stats.unavailable(), stats.evictable};
        }
    }
    return result;
}

void KvCacheManager::_resetIterationPeakNumBlocks(std::optional<CacheLevel> cacheLevel)
{
    if (!cacheLevel.has_value())
    {
        mIterationPeakNumBlocksByCacheLevel = _currentBlockStatsByCacheLevel();
        return;
    }

    PeakBlockStatsByPoolGroup levelStats(mStorage->numPoolGroups(*cacheLevel));
    for (PoolGroupIndex poolGroup{0}; poolGroup < mStorage->numPoolGroups(*cacheLevel); ++poolGroup)
    {
        auto const stats = mStorage->getStatistics(*cacheLevel, poolGroup);
        levelStats[poolGroup] = {stats.available(), stats.unavailable(), stats.evictable};
    }
    mIterationPeakNumBlocksByCacheLevel[*cacheLevel] = std::move(levelStats);
}

void KvCacheManager::_updateIterationPeakNumBlocks()
{
    auto const current = _currentBlockStatsByCacheLevel();
    for (CacheLevel cacheLevel{0}; cacheLevel < current.size(); ++cacheLevel)
    {
        auto& peakLevel = mIterationPeakNumBlocksByCacheLevel[cacheLevel];
        for (PoolGroupIndex poolGroup{0}; poolGroup < current[cacheLevel].size(); ++poolGroup)
        {
            auto& peak = peakLevel[poolGroup];
            auto const& value = current[cacheLevel][poolGroup];
            peak.available = std::max(peak.available, value.available);
            peak.unavailable = std::max(peak.unavailable, value.unavailable);
            peak.evictable = std::max(peak.evictable, value.evictable);
        }
    }
}

PeakBlockStatsByPoolGroup KvCacheManager::getAndResetIterationPeakBlockStats(CacheLevel cacheLevel)
{
    // Exclusive: samples live storage, then resets the accumulator.
    auto const apiLock = lockExclusive();
    _updateIterationPeakNumBlocks();
    PeakBlockStatsByPoolGroup peak = mIterationPeakNumBlocksByCacheLevel.at(cacheLevel);
    _resetIterationPeakNumBlocks(cacheLevel);
    return peak;
}

PeakBlockStatsByCacheLevel KvCacheManager::getAndResetIterationPeakBlockStatsByLevel()
{
    auto const apiLock = lockExclusive();
    _updateIterationPeakNumBlocks();
    PeakBlockStatsByCacheLevel peak = mIterationPeakNumBlocksByCacheLevel;
    _resetIterationPeakNumBlocks();
    return peak;
}

void KvCacheManager::markStatsDirty(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mDirtyStatsKvCacheIds.insert(*kvCacheId);
    }
}

void KvCacheManager::clearStatsDirty(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mDirtyStatsKvCacheIds.erase(*kvCacheId);
    }
}

std::unordered_set<RequestIdType> KvCacheManager::getDirtyStatsKvCacheIds() const
{
    auto const apiLock = lockShared();
    return mDirtyStatsKvCacheIds;
}

void KvCacheManager::markStatsExcluded(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mStatsExcludedKvCacheIds.insert(*kvCacheId);
        clearStatsDirty(kvCacheId);
    }
}

void KvCacheManager::clearStatsExcluded(std::optional<RequestIdType> kvCacheId)
{
    auto const apiLock = lockExclusive();
    if (kvCacheId.has_value())
    {
        mStatsExcludedKvCacheIds.erase(*kvCacheId);
    }
}

bool KvCacheManager::isStatsExcluded(std::optional<RequestIdType> kvCacheId) const
{
    auto const apiLock = lockShared();
    return kvCacheId.has_value() && mStatsExcludedKvCacheIds.find(*kvCacheId) != mStatsExcludedKvCacheIds.end();
}

TypedVec<CacheLevel, CacheTier> KvCacheManager::cacheTierList() const
{
    TypedVec<CacheLevel, CacheTier> result;
    result.reserve(mStorage->mLevels.size());
    for (auto const& lvl : mStorage->mLevels)
    {
        result.push_back(lvl.cacheTier);
    }
    return result;
}

std::vector<BufferId> KvCacheManager::allBufferIds() const
{
    std::vector<BufferId> result;
    result.reserve(mStorage->mBufferAttr.size());
    for (auto const& item : mStorage->mBufferAttr)
        result.push_back(item.first);
    return result;
}

int KvCacheManager::clampMaxSeqLenForMem(int batchSize, int tokenNumUpperBound) const
{
    auto const apiLock = lockShared();
    TLLM_CHECK_DEBUG(batchSize > 0);
    int tokPerBlock = tokensPerBlock();
    PoolGroupIndex numPg = mStorage->numPoolGroups();
    auto const& lcs = mLifeCycles;
    auto const& lcGrouping = mStorage->lifeCycleGrouping(kHotLevel);

    // Remaining slot counts per pool group.
    TypedVec<PoolGroupIndex, SlotCount> remainingSlots(numPg);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        remainingSlots[pgIdx] = mStorage->numSlots(pgIdx);
    }

    // Compute required slot counts per pool group for a given seq_len.
    auto getNumSlots = [&](int seqLen) -> TypedVec<PoolGroupIndex, SlotCount>
    {
        TypedVec<PoolGroupIndex, SlotCount> ret(numPg, 0);
        for (LifeCycleId lifeCycleId{0}; lifeCycleId < lcs.size(); ++lifeCycleId)
        {
            auto staleRange = getStaleRange(lcs[lifeCycleId], seqLen, tokPerBlock);
            int numStaleBlocks = staleRange.end - staleRange.beg;
            int numSlots = divUp(seqLen, tokPerBlock) - numStaleBlocks;
            auto pgIdx = lcGrouping[lifeCycleId];
            ret[pgIdx] += numSlots;
        }
        return ret;
    };

    // Reserve slots for (batch_size - 1) minimal sequences.
    auto minSlots = getNumSlots(1);
    for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
    {
        TLLM_CHECK_DEBUG(minSlots[pgIdx] >= 0);
        SlotCount const reservedSlots = minSlots[pgIdx] * (batchSize - 1);
        remainingSlots[pgIdx] -= reservedSlots;
        if (remainingSlots[pgIdx] < 0)
        {
            return 0;
        }
    }

    auto isEnough = [&](int numBlocks) -> bool
    {
        auto needed = getNumSlots(numBlocks * tokPerBlock);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            TLLM_CHECK_DEBUG(needed[pgIdx] >= 0);
            if (needed[pgIdx] > remainingSlots[pgIdx])
            {
                return false;
            }
        }
        return true;
    };

    if (!isEnough(1))
    {
        return 0;
    }
    int lb = 1;
    int ub = divUp(tokenNumUpperBound, tokPerBlock);
    if (isEnough(ub))
    {
        return tokenNumUpperBound;
    }
    while (lb < ub - 1)
    {
        int mid = (lb + ub) / 2;
        if (isEnough(mid))
        {
            lb = mid;
        }
        else
        {
            ub = mid;
        }
    }
    return std::min(lb * tokPerBlock, tokenNumUpperBound);
}

void KvCacheManager::_adjustLevel(CacheLevel level, size_t quota)
{
    auto const& ratioList = _getTargetRatioList(level);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> const* persistent = nullptr;
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> persistentPages;
    if (mStorage->isLastLevel(level))
    {
        persistentPages = _gatherLastLevelPersistentPages();
        persistent = &persistentPages;
    }
    mStorage->adjustCacheLevel(level, quota, ratioList, persistent);
}

TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> KvCacheManager::_gatherLastLevelPersistentPages() const
{
    CacheLevel lastLevel = mStorage->numCacheLevels() - 1;
    PoolGroupIndex numPg = mStorage->numPoolGroups(lastLevel);
    TypedVec<PoolGroupIndex, std::vector<SharedPtr<Page>>> result(numPg);

    for (KvCache* kvc : mLivingKvCaches)
    {
        TLLM_CHECK_DEBUG(kvc->status() == KvCache::Status::SUSPENDED);
        for (auto const& sb : kvc->blocks())
        {
            for (auto const& beamPages : sb.pages)
            {
                for (LifeCycleId lc{0}; lc < beamPages.size(); ++lc)
                {
                    // Mirrors Python: holder must be _PageHolder type (suspended state).
                    if (blockPageIsNull(beamPages[lc]))
                    {
                        continue;
                    }
                    TLLM_CHECK_DEBUG_WITH_INFO(std::holds_alternative<SharedPtr<PageHolder>>(beamPages[lc]),
                        "Non-null holder must be PageHolder in suspended state");
                    auto const& pg = blockPageGetPage(beamPages[lc]);
                    if (!pg)
                    {
                        continue;
                    }
                    // Mirrors Python assertions for invariant checking.
                    TLLM_CHECK_DEBUG_WITH_INFO(
                        pg->status() == PageStatus::HELD, "Page in suspended KvCache must be HELD");
                    TLLM_CHECK_DEBUG_WITH_INFO((pg->scheduledForEviction() == (pg->cacheLevel != lastLevel)),
                        "Eviction scheduling invariant violated");
                    if (pg->scheduledForEviction())
                    {
                        continue;
                    }
                    PoolGroupIndex pgIdx = mStorage->getPoolGroupIndex(lastLevel, lc);
                    result[pgIdx].push_back(pg);
                }
            }
        }
    }
    return result;
}

// ---- KvCache registry -----------------------------------------------------

void KvCacheManager::registerKvCache(KvCache* kvc)
{
    auto const apiLock = lockExclusive();
    mLivingKvCaches.insert(kvc);
    ++mNumCreatedKvCaches;
}

void KvCacheManager::unregisterKvCache(KvCache* kvc)
{
    auto const apiLock = lockExclusive();
    mLivingKvCaches.erase(kvc);
}

void KvCacheManager::tryUpdateTargetRatios()
{
    if (mNumSampledKvCaches - mLastUpdateNumSampledKvCaches < 100)
        return;
    mLastUpdateNumSampledKvCaches = mNumSampledKvCaches;

    int tokensPerBlock = mConfig.tokensPerBlock;
    int avgReusedLength = static_cast<int>(std::round(mAvgReusedLength.value()));
    int avgCapacity = static_cast<int>(std::round(std::sqrt(mAvgSqrCapacity.value())));
    int avgHistoryLength = static_cast<int>(std::round(std::sqrt(mAvgSqrHistoryLength.value())));
    int avgBeamWidth = std::max(1, static_cast<int>(std::round(mAvgBeamWidth.value())));
    int avgPromptLength = std::max(0, static_cast<int>(std::round(mAvgPromptLength.value())));
    if (avgCapacity > 0)
    {
        auto const lifeCycleRatio = mStorage->ratioFromLength(
            kHotLevel, tokensPerBlock, avgHistoryLength, avgCapacity, avgBeamWidth, avgPromptLength);
        mTargetRatioListHot = mStorage->constrainPoolGroupRatio(mStorage->toPoolGroupRatio(kHotLevel, lifeCycleRatio));
    }
    if (avgReusedLength > 0)
    {
        CacheLevel const coldLevel = mStorage->numCacheLevels() > CacheLevel{1} ? CacheLevel{1} : kHotLevel;
        // The cold tiers hold committed pages, which are canonicalized to beam 0
        // (_commitBlock() truncates the other beams away), so they never carry a
        // beam factor no matter how wide the requests that produced them were.
        auto const lifeCycleRatio
            = mStorage->ratioFromLength(coldLevel, tokensPerBlock, avgReusedLength, avgReusedLength);
        mTargetRatioListCold = mStorage->toPoolGroupRatio(coldLevel, lifeCycleRatio);
    }
}

TypedVec<PoolGroupIndex, float> KvCacheManager::_currentHotRatio() const
{
    return mStorage->getRatioList(kHotLevel);
}

TypedVec<PoolGroupIndex, float> KvCacheManager::_currentColdRatios() const
{
    CacheLevel numLevels = mStorage->numCacheLevels();
    if (numLevels == CacheLevel{1})
    {
        return _currentHotRatio();
    }
    PoolGroupIndex numPg = mStorage->numPoolGroups(CacheLevel{1});
    TypedVec<PoolGroupIndex, float> result(numPg, 0.f);
    for (CacheLevel lvl{1}; lvl < numLevels; ++lvl)
    {
        auto ratios = mStorage->getRatioList(lvl);
        for (PoolGroupIndex pgIdx{0}; pgIdx < numPg; ++pgIdx)
        {
            result[pgIdx] += ratios[pgIdx];
        }
    }
    float denom = static_cast<float>((numLevels - 1).value());
    for (auto& r : result)
    {
        r /= denom;
    }
    return result;
}

// ---- needAdjustment / adjust -----------------------------------------------

TypedVec<PoolGroupIndex, float> const& KvCacheManager::_getTargetRatioList(CacheLevel level) const
{
    return (level == kHotLevel) ? mTargetRatioListHot : mTargetRatioListCold;
}

bool KvCacheManager::_needAdjustment(CacheLevel level) const
{
    auto const& target = _getTargetRatioList(level);
    auto current = (level == kHotLevel) ? _currentHotRatio() : _currentColdRatios();
    constexpr float kThreshold = 1.25f;
    for (PoolGroupIndex pgIdx{0}; pgIdx < target.size() && pgIdx < current.size(); ++pgIdx)
    {
        TLLM_CHECK_DEBUG_WITH_INFO(current[pgIdx] > 0.f && target[pgIdx] > 0.f, "ratios must not be zero");
        float ratio = target[pgIdx] / current[pgIdx];
        if (ratio < 1.f / kThreshold || ratio > kThreshold)
            return true;
    }
    return false;
}

bool KvCacheManager::needAdjustment() const
{
    auto const apiLock = lockShared();
    if (mNumSampledKvCaches < 2000)
        return false;
    double now = nowSeconds();
    if (now - mLastAdjustmentTime < 120.0)
        return false;
    CacheLevel lastLevel = mStorage->numCacheLevels() - 1;
    return _needAdjustment(kHotLevel) || _needAdjustment(lastLevel);
}

void KvCacheManager::adjust()
{
    KVCM2_API_GUARD();
    auto const apiLock = lockExclusive();
    for (KvCache* kvc : mLivingKvCaches)
        TLLM_CHECK_WITH_INFO(kvc->status() == KvCache::Status::SUSPENDED,
            "level adjustment requires every KvCache to be SUSPENDED: _adjustLevel may "
            "defragment, which relocates pages and invalidates indices an ACTIVE cache holds");

    CacheLevel numLevels = mStorage->numCacheLevels();
    for (CacheLevel level{0}; level < numLevels; ++level)
    {
        if (_needAdjustment(level))
            _adjustLevel(level, getQuota(level));
    }
    mLastAdjustmentTime = nowSeconds();
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
