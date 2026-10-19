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

#include "kv_cache_manager_v2/batchedPageCopy.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/indexConversionKernels.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <numeric>

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

Batch::Batch(KvCacheManager& manager, CUstream stream, int maxBatchSize, int maxBlocksPerSeq,
    std::optional<PageIndexMode> mode, std::vector<IndexTableRequest> indexTables,
    std::optional<int> maxTotalInputTokens)
    : mManager(&manager)
    , mCudaStream(stream)
    , mMaxTotalInputTokens(maxTotalInputTokens)
{
    if (maxTotalInputTokens.has_value() && *maxTotalInputTokens <= 0)
    {
        throw std::invalid_argument("Batch: maxTotalInputTokens must be positive when given");
    }
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

    // Prefer SHARED: one table per life cycle rather than one per layer. The manager
    // answers from configuration alone, so this cannot drift with what a member is doing.
    mIndexModes = TypedVec<LifeCycleId, PageIndexMode>(mNumLayerGroups, PageIndexMode::PER_LAYER);
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        mIndexModes[lcId]
            = mode.value_or(manager.supportsIndexMode(PageIndexMode::SHARED, lcId) == std::optional<bool>(true)
                    ? PageIndexMode::SHARED
                    : PageIndexMode::PER_LAYER);
    }

    // Page tables and watermarks are published together, so a kernel cannot read one step's
    // table beside another step's lengths -- an out-of-range read rather than a stale
    // answer.
    auto const elements = _totalElements();
    mStaging = std::make_unique<HostMem>(_totalBytes());
    auto* staging = reinterpret_cast<int32_t*>(mStaging->address());
    // A row that no member occupies must still read as "no such block" rather than as slot
    // zero, since the fetch has no other way to tell them apart.
    std::fill(staging, staging + _historyOffset(), kBadPageIndex.value());
    // Watermarks of a free row are zero: a request with nothing eligible.
    std::fill(staging + _historyOffset(), staging + elements, 0);

    // Views start default-constructed, so an unpublished batch reads as having no caches.
    auto* views = reinterpret_cast<SparseCacheView*>(reinterpret_cast<std::byte*>(staging) + _viewsOffsetBytes());
    auto const numViews = static_cast<size_t>(mNumLayerGroups.value()) * static_cast<size_t>(mMaxBatchSize);
    std::fill(views, views + numViews, SparseCacheView{});

    // The staging mirrors the device block exactly, so publishing it is one linear copy.
    // It also doubles what a Batch costs in device memory, which is the price of letting
    // the host-to-device crossing overlap the previous step; a caller that already holds a
    // buffer of this shape could lend it instead.
    mDevice = allocateDeviceMemory(_deviceBytes());
    mDeviceStaging = allocateDeviceMemory(_deviceBytes());
    auto const device = reinterpret_cast<CUdeviceptr>(mDevice.get());
    auto const deviceStaging = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());

    for (auto const base : {device, deviceStaging})
    {
        cuCheck(cuMemcpyHtoD(base, staging, _totalBytes()));
    }

    _allocateStepScratch();
    _allocateFlushRanges();

    // After the modes: a request's offsets depend on where its life cycle puts the base.
    _buildIndexTables(indexTables);
    _checkSparseRunsAreAddressable();
    _allocateIndexTables();

    // Last: a descriptor names where in the index tables a buffer's converted indices sit.
    _allocateCopyDesc();
}

void Batch::_allocateCopyDesc()
{
    // One descriptor per buffer, serving both directions, for every life cycle. Small
    // enough to share an allocation and a lifetime.
    std::vector<SparseCopyDesc> host;
    std::vector<size_t> offset(static_cast<size_t>(mNumLayerGroups.value()), 0);

    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth == 0)
        {
            continue;
        }
        auto const runs = mManager->sparseCopyTemplates(lcId);
        auto const plan = mManager->sparseBufferPlan(lcId);
        scratch.numBuffers = static_cast<int32_t>(plan.size());
        offset[static_cast<size_t>(lcId.value())] = host.size();

        for (auto const& run : runs)
        {
            // A run is consecutive buffers of one pool and, since buffers are ordered
            // (layer, pool), of one layer. Its first buffer names the index table every
            // buffer of the run is relative to -- the converted index carries each one's
            // own offset within the coalesced page.
            auto const& first = plan[static_cast<size_t>(run.bufferBase)].id;
            auto const table = sparseIndexRun(first.layerId, first.role);

            for (int32_t i = 0; i < run.numBuffers; ++i)
            {
                auto const buffer = run.bufferBase + i;
                SparseCopyDesc desc;
                desc.hostBase = static_cast<CUdeviceptr>(table.base);
                desc.hostPageStride = run.hostPageStride;
                // Pre-offset to this buffer's entry, so the kernel indexes the row alone.
                desc.pageTable
                    = reinterpret_cast<int32_t const*>(table.table) + static_cast<ptrdiff_t>(i) * table.entryStride;
                desc.pageTableStride = static_cast<uint32_t>(table.rowStride);
                desc.numBlocks = static_cast<uint32_t>(table.rowStride);
                // The buffer's own slot in the pool, folded in so the kernel need not read
                // it through a view it loads per work item.
                desc.deviceBase = run.deviceBase
                    + static_cast<uint64_t>(
                          mManager->sparseArena(lcId).shape().bufferSlotInPool[static_cast<size_t>(buffer)])
                        * run.cellsPerBuffer * run.bufferBytes;
                desc.leaseStride = static_cast<uint32_t>(run.cellsPerLease) * run.bufferBytes;
                desc.cellBytes = run.bufferBytes;
                desc.layout = run.layout;
                desc.width = run.width;

                auto const width = static_cast<uint64_t>(sparseCopyWidthBytes(desc.width));
                TLLM_CHECK_WITH_INFO(desc.hostBase % width == 0 && desc.deviceBase % width == 0,
                    "A sparse buffer's bases must be aligned to the %llu bytes its layout copies at a time; host "
                    "base %llu, device base %llu",
                    static_cast<unsigned long long>(width), static_cast<unsigned long long>(desc.hostBase),
                    static_cast<unsigned long long>(desc.deviceBase));
                host.push_back(desc);
            }
        }
    }

    if (host.empty())
    {
        return;
    }
    mCopyDesc = allocateDeviceMemory<SparseCopyDesc>(host.size());
    auto const* device = mCopyDesc.get();
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(device), host.data(), sizeof(SparseCopyDesc) * host.size()));
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth != 0)
        {
            scratch.copyDesc = device + offset[static_cast<size_t>(lcId.value())];
        }
    }
}

int32_t Batch::_flushCapacity(LifeCycleId lifeCycle) const
{
    // A row advancing `t` tokens crosses at most `t / unitTokens + 1` units -- the `+1` for
    // the partial unit its watermark starts inside -- so the batch crosses at most
    // `maxTotalInputTokens / unitTokens + maxBatchSize`. The bound is an aggregate one, and
    // usable only because the list is compacted: indexed by row, every row's slice would
    // have to allow for that row taking the whole budget.
    int const perPage = mManager->sparseCacheUnitsPerPage(lifeCycle);
    int const unitTokens = perPage > 0 ? mManager->config().tokensPerBlock / perPage : 0;
    if (mMaxTotalInputTokens.has_value() && unitTokens > 0)
    {
        return *mMaxTotalInputTokens / unitTokens + mMaxBatchSize;
    }
    // Without a budget, a step is assumed to advance capacity by the in-flight headroom in
    // every row at once, which is what a decode step does.
    return mMaxBatchSize * std::max(1, mManager->sparseCacheMaxInFlightUnits(lifeCycle));
}

void Batch::_allocateStepScratch()
{
    mScratch = TypedVec<LifeCycleId, SparseStepScratch>(mNumLayerGroups, SparseStepScratch{});

    // Sized up front from configuration, so no step has to allocate. Laid out as one block
    // because a per-group allocation would be a dozen tiny ones serving the same lifetime.
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    size_t total = 0;
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        int const width = mManager->sparseCacheSelectionWidth(lcId);
        if (mManager->sparseCacheMaxUnits(lcId) <= 0 || width <= 0)
        {
            continue;
        }
        auto& scratch = mScratch[lcId];
        scratch.selectionWidth = width;
        scratch.maxInFlightUnits = mManager->sparseCacheMaxInFlightUnits(lcId);
        scratch.numLayers = mManager->sparseCacheLayerCount(lcId);
        auto const layers = static_cast<size_t>(scratch.numLayers);
        // One list for the fetch, holding the worst case of every row missing its whole
        // selection; one per layer for the write-back, each holding the units the step's
        // token budget can carry across a watermark.
        scratch.fetchWork.capacity = static_cast<int32_t>(rows) * width;
        scratch.fetchWork.numLists = 1;
        scratch.flushWork.capacity = _flushCapacity(lcId);
        scratch.flushWork.numLists = scratch.numLayers;
        // A counter per layer even though the fetch's list is one: the list is reused from
        // layer to layer, because a layer's copy has consumed it before the next layer's
        // resolve overwrites it, but the counter has to read zero when that resolve starts.
        // Giving each layer its own lets the whole step's counters be cleared in one go,
        // rather than one clear per layer standing between every resolve and whatever ran
        // before it. Per-layer *lists* would cost the layer count times the list.
        mCounterElements += 2 * layers;
        auto const fetchEntries = 3 * static_cast<size_t>(scratch.fetchWork.capacity);
        auto const flushEntries = layers * 3 * static_cast<size_t>(scratch.flushWork.capacity);
        total += rows * (static_cast<size_t>(width) + 1 + static_cast<size_t>(scratch.maxInFlightUnits));
        total += fetchEntries + flushEntries + rows * layers;
    }
    if (total == 0)
    {
        return;
    }

    mStepScratch = allocateDeviceMemory<int32_t>(total + mCounterElements);
    // Its own allocation: the step's clear covers mCounters, and these have to survive it.
    mHitMiss = allocateDeviceMemory<SparseHitMiss>(1);
    cuCheck(cuMemsetD32(reinterpret_cast<CUdeviceptr>(mHitMiss.get()), 0U, sizeof(SparseHitMiss) / sizeof(uint32_t)));

    // The counters lead the block so that one clear covers every life cycle's.
    auto* counter = mStepScratch.get();
    mCounters = counter;
    auto* cursor = counter + mCounterElements;
    auto carve = [&cursor](size_t count)
    {
        auto* ptr = cursor;
        cursor += count;
        return ptr;
    };
    auto const carveCounters = [&counter](size_t count)
    {
        auto* ptr = counter;
        counter += count;
        return ptr;
    };
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth == 0)
        {
            continue;
        }
        auto const width = static_cast<size_t>(scratch.selectionWidth);
        scratch.outSlot = carve(rows * width);
        scratch.missCount = carve(rows);
        scratch.inFlightSlot = carve(rows * static_cast<size_t>(scratch.maxInFlightUnits));

        auto const carveList = [&carve](SparseWorkList& work)
        {
            auto const entries = static_cast<size_t>(work.capacity) * static_cast<size_t>(work.numLists);
            work.row = carve(entries);
            work.ordinal = carve(entries);
            work.slot = carve(entries);
        };
        carveList(scratch.fetchWork);
        carveList(scratch.flushWork);
        auto const layerCount = static_cast<size_t>(scratch.numLayers);
        // The fetch takes one counter per layer against its single list, so a layer names
        // its own by offsetting this; the write-back's counters match its lists one to one.
        scratch.fetchWork.total = carveCounters(layerCount);
        scratch.flushWork.total = carveCounters(layerCount);
        scratch.hitMiss = mHitMiss.get();

        auto const layers = static_cast<size_t>(scratch.numLayers);
        scratch.flushCount = carve(rows * layers);
        // -1 so that "no write-back has run" stays distinguishable from "one ran and found
        // nothing to move", which is what lets a test tell the two apart.
        cuCheck(cuMemsetD32(reinterpret_cast<CUdeviceptr>(scratch.flushCount), 0xFFFFFFFFU, rows * layers));
    }
}

SparseStepScratch const& Batch::sparseStepScratch(LayerGroupId layerGroupId) const
{
    LifeCycleId const lcId{layerGroupId.value()};
    if (lcId >= mScratch.size())
    {
        throw std::out_of_range("Batch::sparseStepScratch: no such layer group");
    }
    return mScratch[lcId];
}

Batch::~Batch()
{
    // Both buffers are about to go, and transfers on the upload stream still name them.
    _drain();
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            for (auto const& kvCache : mRows)
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

PageIndexMode Batch::indexMode(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::indexMode: unknown layer group");
    }
    return mIndexModes[layerGroupId];
}

void Batch::_buildIndexTables(std::vector<IndexTableRequest> const& requests)
{
    // The offset of a buffer relative to a chosen base, in page-index units. Mirrors the
    // runtime's kv_offset: a base difference divided by the page stride, which must divide
    // exactly or the two buffers do not sit a whole number of pages apart.
    auto pageOffset = [this](BufferId const& buffer, MemAddress base)
    {
        auto const bufferBase = mManager->getMemPoolBaseAddress(buffer.layerId, buffer.role, PageIndexMode::SHARED);
        auto const stride = static_cast<int64_t>(mManager->getPageStride(buffer.layerId, buffer.role));
        TLLM_CHECK_WITH_INFO(stride > 0, "Page stride must be positive to offset a buffer within its slot");
        auto const delta = static_cast<int64_t>(bufferBase) - static_cast<int64_t>(base);
        TLLM_CHECK_WITH_INFO(delta % stride == 0,
            "Buffers sharing an index table must sit a whole number of pages apart; they cannot share one");
        return static_cast<int>(delta / stride);
    };

    for (auto const& request : requests)
    {
        TLLM_CHECK_WITH_INFO(!request.buffers.empty(), "An index table request names no buffers");

        auto const& first = request.buffers.front();
        auto const lifeCycle = mManager->bufferAttr(first.layerId, first.role).lifeCycleId;
        // Substitution replaces one layer's page, so every layer of the life cycle can no
        // longer read one table. SHARED is exactly the claim that they can.
        bool const mayHoldScratch = !mManager->supportsIndexMode(PageIndexMode::SHARED, lifeCycle).has_value();
        TLLM_CHECK_WITH_INFO(!mayHoldScratch || mIndexModes[lifeCycle] == PageIndexMode::PER_LAYER,
            "A life cycle that can hold scratch slots needs PER_LAYER index tables; its layers stop sharing one "
            "set of converted indices as soon as a scratch slot is substituted");
        auto const scale = mManager->getPageIndexScale(first.layerId, first.role);
        auto const mode = mIndexModes[lifeCycle];

        // Under SHARED the base carries the per-layer offset, so the request's own offsets
        // are relative to its first buffer. Under PER_LAYER the base is the pool group's
        // and carries none, so every offset is absolute within the slot.
        auto const base = mode == PageIndexMode::SHARED
            ? mManager->getMemPoolBaseAddress(first.layerId, first.role, PageIndexMode::SHARED)
            : mManager->getMemPoolBaseAddress(first.layerId, first.role, PageIndexMode::PER_LAYER);

        std::vector<int> offsets;
        offsets.reserve(request.buffers.size());
        for (auto const& buffer : request.buffers)
        {
            auto const& attr = mManager->bufferAttr(buffer.layerId, buffer.role);
            // The reference converter emits `expansion` entries per block. Nothing in the
            // tree sets tokensPerBlockOverride, and every attention backend refuses an
            // expansion of anything but one, so this follows them rather than building a
            // path no caller can reach.
            TLLM_CHECK_WITH_INFO(attr.expansion == 1,
                "Index conversion does not implement page index expansion, so a buffer with a tokens-per-block "
                "override cannot have converted index tables");
            TLLM_CHECK_WITH_INFO(attr.lifeCycleId == lifeCycle,
                "An index table request spans life cycles; their slot ids differ, so no one table can serve it");
            TLLM_CHECK_WITH_INFO(mManager->getPageIndexScale(buffer.layerId, buffer.role) == scale,
                "An index table request spans page index scales; their converted indices differ, so no one table "
                "can serve it");
            offsets.push_back(pageOffset(buffer, base));
        }

        // Requests coincide when their entries do. Under PER_LAYER the table is per layer,
        // so requests for different layers never merge however alike their entries look.
        auto const sameTable = [&](IndexTable const& table)
        {
            if (table.lifeCycle != lifeCycle || table.scale != scale || table.concatOffsets != offsets)
            {
                return false;
            }
            return mode == PageIndexMode::SHARED || table.served.front().buffers.front().layerId == first.layerId;
        };

        auto found = std::find_if(mIndexTables.begin(), mIndexTables.end(), sameTable);
        if (found == mIndexTables.end())
        {
            mIndexTables.push_back(IndexTable{{}, lifeCycle, scale, offsets});
            found = std::prev(mIndexTables.end());
        }
        IndexTableId const tableId{static_cast<int>(std::distance(mIndexTables.begin(), found))};
        found->served.push_back(request);

        for (size_t slot = 0; slot < request.buffers.size(); ++slot)
        {
            auto const& buffer = request.buffers[slot];
            auto const key = std::make_pair(buffer.layerId, buffer.role);
            BufferSlot const resolved{tableId, static_cast<int>(slot), base};
            auto const [it, inserted] = mBufferSlots.emplace(key, resolved);
            TLLM_CHECK_WITH_INFO(inserted
                    || (it->second.table == resolved.table && it->second.entry == resolved.entry
                        && it->second.base == resolved.base),
                "A buffer appears in two index table requests that resolved differently");
        }
    }
}

void Batch::_allocateFlushRanges()
{
    mFlushedHistory.assign(static_cast<size_t>(mMaxBatchSize), 0);

    bool anySparse = false;
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        anySparse = anySparse || mManager->sparseCacheMaxUnits(lcId) > 0;
    }
    if (!anySparse)
    {
        return;
    }

    auto const elements = 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mNumLayerGroups.value());
    mFlushRanges = allocateDeviceMemory<int32_t>(elements);
    // An empty range asks for nothing, which is what a row that never flushes should read.
    cuCheck(cuMemsetD32(reinterpret_cast<CUdeviceptr>(mFlushRanges.get()), 0, elements));
    mFlushRangeStaging.assign(elements, 0);
}

int32_t const* Batch::flushRangeRow(LifeCycleId lifeCycle) const
{
    auto* base = reinterpret_cast<int32_t*>(mFlushRanges.get());
    return base + 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(lifeCycle.value());
}

int Batch::flushedHistory(int row) const
{
    if (row < 0 || row >= mMaxBatchSize)
    {
        throw std::out_of_range("Batch::flushedHistory: row is beyond max_batch_size");
    }
    return mFlushedHistory[static_cast<size_t>(row)];
}

void Batch::flushSparseUnits(std::vector<int> const& newHistoryLengths)
{
    if (newHistoryLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch::flushSparseUnits: one history length per row");
    }
    if (!mFlushRanges)
    {
        // No sparse life cycle, so nothing is ever in flight and the record is vacuous.
        for (size_t row = 0; row < mRows.size(); ++row)
        {
            mFlushedHistory[row] = newHistoryLengths[row];
        }
        return;
    }

    // Every row is checked before any work is issued, so a move no reservation could cover
    // leaves the batch as it was.
    bool anyWork = false;
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
        if (mManager->sparseCacheMaxUnits(lcId) <= 0 || unitsPerPage <= 0)
        {
            continue;
        }
        int const unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
        auto const base = static_cast<size_t>(2 * mMaxBatchSize * lcId.value());

        // Bounded in aggregate rather than per row, which is what the compacted list allows
        // and what in-flight batching needs: one prefilling row may cross the whole step's
        // worth of units while the rest cross none.
        int64_t crossed = 0;

        for (size_t row = 0; row < mRows.size(); ++row)
        {
            auto const& kvCache = mRows[row];
            int fromUnit = 0;
            int toUnit = 0;
            if (kvCache != nullptr)
            {
                if (newHistoryLengths[row] < kvCache->historyLength())
                {
                    throw std::invalid_argument("Batch::flushSparseUnits: history length cannot be decreased");
                }
                // Clamped to the capacity this row starts from: a unit at or above it was
                // never in flight, so it has no cell and nothing to write back.
                fromUnit = kvCache->historyLength() / unitTokens;
                toUnit = std::min(newHistoryLengths[row], kvCache->capacity()) / unitTokens;
                toUnit = std::max(toUnit, fromUnit);

                crossed += toUnit - fromUnit;
                anyWork = anyWork || toUnit > fromUnit;
            }
            mFlushRangeStaging[base + row] = fromUnit;
            mFlushRangeStaging[base + static_cast<size_t>(mMaxBatchSize) + row] = toUnit;
        }

        if (crossed > mScratch[lcId].flushWork.capacity)
        {
            throw std::invalid_argument(
                "Batch::flushSparseUnits: the batch crossed more units than the step's token budget allows, so "
                "the write-back list could not name them all; construct the Batch with maxTotalInputTokens");
        }
    }

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        mFlushedHistory[row] = newHistoryLengths[row];
    }
    if (!anyWork)
    {
        // Nothing crossed anywhere, so the ranges on the device are stale but unread.
        return;
    }

    auto* const upload = mUploadStream.handle();
    if (!mFlushRead.isClosed())
    {
        // The previous step's write-back read these same ranges; its reads must retire
        // before the copy below overwrites them.
        mFlushRead.waitInStream(reinterpret_cast<CudaStream>(upload));
        mFlushRead.close();
    }
    detail::copyFlushRangesToDevice(reinterpret_cast<CUdeviceptr>(mFlushRanges.get()), mFlushRangeStaging.data(),
        mFlushRangeStaging.size(), upload);
    auto staged = mUploadStream.recordEvent();
    staged.waitInStream(reinterpret_cast<CudaStream>(mCudaStream));

    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        if (mManager->sparseCacheMaxUnits(lcId) <= 0)
        {
            continue;
        }
        mManager->writeBackSparseUnits(*this, lcId, mCudaStream);
    }
    mFlushRead = CachedCudaEvent{reinterpret_cast<CudaStream>(mCudaStream)};
}

void Batch::_checkSparseRunsAreAddressable() const
{
    for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
    {
        auto const plan = mManager->sparseBufferPlan(lcId);
        if (plan.empty())
        {
            continue;
        }
        for (size_t begin = 0; begin < plan.size();)
        {
            size_t end = begin;
            while (end < plan.size() && plan[end].poolIndex == plan[begin].poolIndex
                && plan[end].layerRow == plan[begin].layerRow)
            {
                ++end;
            }

            auto const slotOf = [this](SparseBufferPlan const& entry)
            {
                auto const found = mBufferSlots.find(std::make_pair(entry.id.layerId, entry.id.role));
                TLLM_CHECK_WITH_INFO(found != mBufferSlots.end(),
                    "A sparse buffer has no index table request; the fetch reads converted indices, so every "
                    "sparse buffer needs one");
                return found->second;
            };

            auto const first = slotOf(plan[begin]);
            for (size_t i = begin + 1; i < end; ++i)
            {
                auto const slot = slotOf(plan[i]);
                TLLM_CHECK_WITH_INFO(slot.table == first.table && slot.base == first.base,
                    "Sparse buffers sharing a pool are copied in one launch, from one base pointer, so they must "
                    "share one index table request");
                TLLM_CHECK_WITH_INFO(slot.entry == first.entry + static_cast<int>(i - begin),
                    "Sparse buffers sharing a pool must sit at consecutive entries of their request, in the order "
                    "the pool holds them");
            }
            begin = end;
        }
    }
}

void Batch::_allocateIndexTables()
{
    if (mIndexTables.empty())
    {
        return;
    }

    // Scratch-capable life cycles sort last so that one launch covers the plain tables and
    // one the substituting ones, rather than a kernel branching per table.
    auto const mayHoldScratch = [this](IndexTable const& table)
    { return !mManager->supportsIndexMode(PageIndexMode::SHARED, table.lifeCycle).has_value(); };

    std::vector<size_t> order(mIndexTables.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_partition(
        order.begin(), order.end(), [&](size_t index) { return !mayHoldScratch(mIndexTables[index]); });

    std::vector<IndexTable> sorted;
    std::vector<int> remap(mIndexTables.size());
    sorted.reserve(mIndexTables.size());
    for (size_t position = 0; position < order.size(); ++position)
    {
        remap[order[position]] = static_cast<int>(position);
        sorted.push_back(std::move(mIndexTables[order[position]]));
    }
    mIndexTables = std::move(sorted);
    for (auto& [key, slot] : mBufferSlots)
    {
        slot.table = IndexTableId{remap[static_cast<size_t>(slot.table.value())]};
    }
    mNumPlainTables = static_cast<int>(std::count_if(
        mIndexTables.begin(), mIndexTables.end(), [&](IndexTable const& table) { return !mayHoldScratch(table); }));

    // Mirrors the runtime's bound: the most slots a request could need for one life cycle.
    for (size_t index = static_cast<size_t>(mNumPlainTables); index < mIndexTables.size(); ++index)
    {
        auto const& table = mIndexTables[index];
        auto const& first = table.served.front().buffers.front();
        auto const pages = mManager->getPageIndexConverter(first.layerId, first.role).scratchPagesPerBlock;
        auto const slots = (static_cast<int64_t>(mMaxBlocksPerSeq) * pages + table.scale - 1) / table.scale;
        mMaxScratchSlots = std::max<int>(mMaxScratchSlots, static_cast<int>(std::max<int64_t>(1, slots)));
    }

    if (mMaxScratchSlots > 0)
    {
        auto const elements = _scratchElements();
        mScratchStaging = std::make_unique<HostMem>(sizeof(int32_t) * elements);
        auto* host = reinterpret_cast<int32_t*>(mScratchStaging->address());
        // An empty range leaves every block on the plain path, which is what a member
        // holding no scratch should look like.
        std::fill(host, host + elements, 0);
        mScratchDevice = allocateDeviceMemory<int32_t>(elements);
        cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mScratchDevice.get()), host, sizeof(int32_t) * elements));
    }

    auto const rowBlocks = static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
    size_t elements = 0;
    for (auto& table : mIndexTables)
    {
        table.elementOffset = elements;
        elements += table.concatOffsets.size() * rowBlocks;
        mMaxTableEntries = std::max(mMaxTableEntries, static_cast<int>(table.concatOffsets.size()));
    }

    mIndexTableMemory = allocateDeviceMemory<int32_t>(elements);
    auto const memory = reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get());
    // A table a publish has not reached yet must read as absent rather than as slot zero.
    cuCheck(cuMemsetD32(memory, static_cast<unsigned int>(kBadPageIndex.value()), elements));

    // The descriptors and the offsets they point at share one allocation: the offsets are a
    // handful of ints per table and would otherwise be an allocation each.
    size_t offsetElements = 0;
    for (auto const& table : mIndexTables)
    {
        offsetElements += table.concatOffsets.size();
    }
    size_t const descBytes = sizeof(IndexTableDesc) * mIndexTables.size();
    size_t const offsetsBytes = sizeof(int32_t) * offsetElements;

    mIndexTableDescs = allocateDeviceMemory(descBytes + offsetsBytes);
    auto const descs = reinterpret_cast<CUdeviceptr>(mIndexTableDescs.get());

    std::vector<int32_t> offsetHost;
    offsetHost.reserve(offsetElements);
    std::vector<IndexTableDesc> descHost;
    descHost.reserve(mIndexTables.size());

    auto const offsetsBase = descs + descBytes;
    auto const rawBase = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());
    for (auto const& table : mIndexTables)
    {
        IndexTableDesc desc;
        // The raw slot ids of this table's life cycle, in the staging the upload filled.
        desc.raw = reinterpret_cast<int32_t const*>(
            rawBase + sizeof(int32_t) * _rowElements() * static_cast<size_t>(table.lifeCycle.value()));
        desc.converted = reinterpret_cast<int32_t*>(memory + sizeof(int32_t) * table.elementOffset);
        desc.offsets = reinterpret_cast<int32_t const*>(offsetsBase + sizeof(int32_t) * offsetHost.size());
        desc.numEntries = static_cast<int32_t>(table.concatOffsets.size());
        desc.scale = table.scale;
        if (static_cast<int>(descHost.size()) >= mNumPlainTables)
        {
            auto const row = static_cast<size_t>(mMaxBatchSize);
            auto const lc = static_cast<size_t>(table.lifeCycle.value());
            auto const begs = reinterpret_cast<CUdeviceptr>(mScratchDevice.get());
            auto const groups = static_cast<size_t>(mNumLayerGroups.value());
            desc.scratchBegs = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * lc * row);
            desc.scratchEnds = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * (groups * row + lc * row));
            desc.scratchSlots = reinterpret_cast<int32_t const*>(
                begs + sizeof(int32_t) * (2 * groups * row + lc * row * static_cast<size_t>(mMaxScratchSlots)));
            desc.scratchSlotStride = mMaxScratchSlots;
            auto const& firstBuffer = table.served.front().buffers.front();
            desc.scratchPagesPerBlock
                = mManager->getPageIndexConverter(firstBuffer.layerId, firstBuffer.role).scratchPagesPerBlock;
        }
        descHost.push_back(desc);
        offsetHost.insert(offsetHost.end(), table.concatOffsets.begin(), table.concatOffsets.end());
    }

    cuCheck(cuMemcpyHtoD(descs, descHost.data(), descBytes));
    cuCheck(cuMemcpyHtoD(offsetsBase, offsetHost.data(), offsetsBytes));
}

void Batch::_convertIndices(CUstream stream)
{
    if (mIndexTables.empty())
    {
        return;
    }
    auto const* const descs = reinterpret_cast<IndexTableDesc const*>(mIndexTableDescs.get());
    launchIndexConversion(descs, mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), stream);
    launchIndexConversionWithScratch(descs + mNumPlainTables,
        static_cast<int32_t>(mIndexTables.size()) - mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), stream);
}

CUdeviceptr Batch::indexTableAddress(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::indexTableAddress: unknown table");
    }
    return reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get())
        + sizeof(int32_t) * mIndexTables[static_cast<size_t>(tableId.value())].elementOffset;
}

ConvertedIndexTable Batch::convertedIndexTable(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::convertedIndexTable: unknown table");
    }
    ConvertedIndexTable view;
    view.data = indexTableAddress(tableId);
    // A table's entries are its requests' buffers: the offsets it converts with, one each.
    view.entries = static_cast<int32_t>(mIndexTables[static_cast<size_t>(tableId.value())].concatOffsets.size());
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

Batch::SparseIndexRun Batch::sparseIndexRun(LayerId layerId, DataRole const& role) const
{
    auto const [tableId, entry] = indexTableSlot(layerId, role);
    SparseIndexRun run;
    run.rowStride = mMaxBlocksPerSeq;
    run.entryStride = mMaxBatchSize * mMaxBlocksPerSeq;
    run.table = indexTableAddress(tableId) + sizeof(int32_t) * static_cast<size_t>(entry) * run.entryStride;
    run.base = indexTableBase(layerId, role);
    return run;
}

size_t Batch::_scratchElements() const noexcept
{
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mNumLayerGroups.value());
    // [begs][ends][slots], each indexed by life cycle then row.
    return 2 * groups * rows + groups * rows * static_cast<size_t>(mMaxScratchSlots);
}

void Batch::_stageScratch()
{
    if (!mScratchStaging)
    {
        return;
    }
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mNumLayerGroups.value());
    auto* const begs = reinterpret_cast<int32_t*>(mScratchStaging->address());
    auto* const ends = begs + groups * rows;
    auto* const slots = ends + groups * rows;

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
        {
            auto const flat = static_cast<size_t>(lcId.value()) * rows + row;
            auto* const rowSlots = slots + flat * static_cast<size_t>(mMaxScratchSlots);

            // A free row, a suspended member, or one holding no scratch here all leave an
            // empty range, which the kernel reads as "every block takes the plain path".
            auto const desc = kvCache != nullptr && kvCache->isActive() ? kvCache->getScratchDesc(lcId) : std::nullopt;
            if (!desc.has_value())
            {
                begs[flat] = 0;
                ends[flat] = 0;
                continue;
            }
            begs[flat] = desc->range.beg.value();
            ends[flat] = desc->range.end.value();
            TLLM_CHECK_WITH_INFO(desc->slotIds.size() <= static_cast<size_t>(mMaxScratchSlots),
                "A request holds more scratch slots than the batch was sized for");
            std::copy(desc->slotIds.begin(), desc->slotIds.end(), rowSlots);
        }
    }
}

IndexTableId Batch::numIndexTables() const noexcept
{
    return IndexTableId{static_cast<int>(mIndexTables.size())};
}

std::vector<IndexTableRequest> const& Batch::indexTable(IndexTableId tableId) const
{
    if (tableId.value() < 0 || tableId >= numIndexTables())
    {
        throw std::out_of_range("Batch::indexTable: unknown table");
    }
    return mIndexTables[static_cast<size_t>(tableId.value())].served;
}

std::pair<IndexTableId, int> Batch::indexTableSlot(LayerId layerId, DataRole const& role) const
{
    auto const found = mBufferSlots.find(std::make_pair(layerId, role));
    if (found == mBufferSlots.end())
    {
        throw std::out_of_range("Batch::indexTableSlot: no index table request named this buffer");
    }
    return {found->second.table, found->second.entry};
}

MemAddress Batch::indexTableBase(LayerId layerId, DataRole const& role) const
{
    auto const found = mBufferSlots.find(std::make_pair(layerId, role));
    if (found == mBufferSlots.end())
    {
        throw std::out_of_range("Batch::indexTableBase: no index table request named this buffer");
    }
    return found->second.base;
}

size_t Batch::_deviceBytes() const noexcept
{
    return _totalBytes();
}

void Batch::_drain() noexcept
{
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            waitPublished();
            if (!mStagingRead.isClosed())
            {
                mStagingRead.synchronize();
                mStagingRead.close();
            }
        });
}

size_t Batch::_historyOffset() const noexcept
{
    return _rowElements() * static_cast<size_t>(mNumLayerGroups.value());
}

size_t Batch::_capacityOffset() const noexcept
{
    return _historyOffset() + static_cast<size_t>(mMaxBatchSize);
}

size_t Batch::_totalElements() const noexcept
{
    return _capacityOffset() + static_cast<size_t>(mMaxBatchSize);
}

size_t Batch::_viewsOffsetBytes() const noexcept
{
    // The views hold pointers, so the region starts at their alignment rather than wherever
    // the int32 arrays happen to end.
    size_t const bytes = sizeof(int32_t) * _totalElements();
    size_t const align = alignof(SparseCacheView);
    return (bytes + align - 1) / align * align;
}

size_t Batch::_totalBytes() const noexcept
{
    return _viewsOffsetBytes()
        + sizeof(SparseCacheView) * static_cast<size_t>(mNumLayerGroups.value()) * static_cast<size_t>(mMaxBatchSize);
}

void Batch::_stageRowState()
{
    auto* bytes = reinterpret_cast<std::byte*>(mStaging->address());
    auto* base = reinterpret_cast<int32_t*>(bytes);
    auto* history = base + _historyOffset();
    auto* capacity = base + _capacityOffset();
    auto* views = reinterpret_cast<SparseCacheView*>(bytes + _viewsOffsetBytes());

    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        history[row] = kvCache == nullptr ? 0 : kvCache->historyLength();
        capacity[row] = kvCache == nullptr ? 0 : kvCache->capacity();

        for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
        {
            auto& slot = views[static_cast<size_t>(lg.value()) * mRows.size() + row];
            // A free row, or a member holding no reservation for this group, leaves the
            // default view: null pointers the kernel reads as "nothing to resolve here".
            slot = kvCache == nullptr || lg >= kvCache->mSparseCache.size() ? SparseCacheView{}
                                                                            : kvCache->mSparseCache[lg].view();
        }
    }
}

void Batch::_seedRowWatermarks(int row, int historyLength, int capacity)
{
    auto* base = reinterpret_cast<int32_t*>(mStaging->address());
    auto const index = static_cast<size_t>(row);
    base[_historyOffset() + index] = historyLength;
    base[_capacityOffset() + index] = capacity;
}

int32_t* Batch::_stagingRow(LayerGroupId layerGroupId, int row) const
{
    auto* base = reinterpret_cast<int32_t*>(mStaging->address());
    return base + static_cast<int64_t>(_rowElements()) * layerGroupId.value()
        + static_cast<int64_t>(mMaxBlocksPerSeq) * row;
}

int Batch::addKvCache(KvCache& kvCache, int requestedRow)
{
    if (kvCache.mBatch != nullptr)
    {
        throw LogicError("A KvCache belongs to at most one Batch");
    }
    if (kvCache.beamWidth() != BeamIndex{1})
    {
        // Only beam 0 is mirrored below, so a wider request would publish one beam's pages
        // and have every beam read them. Refusing beats a wrong answer nothing would notice.
        throw LogicError("Batch does not support beam search yet; a member must have beam width 1");
    }
    if (!kvCache.isActive())
    {
        // Membership means the member is being stepped, which is what lets its arena lease,
        // its row and its active state be one thing rather than three that can disagree.
        throw LogicError("A KvCache must be resumed before it joins a Batch");
    }
    // Binding a row writes it, so the last publish must be done reading it.
    waitPublished();
    int row = requestedRow;
    if (row < 0)
    {
        auto const free = std::find(mRows.begin(), mRows.end(), nullptr);
        if (free == mRows.end())
        {
            throw LogicError("Batch is full; it was sized for max_batch_size rows");
        }
        row = static_cast<int>(std::distance(mRows.begin(), free));
    }
    else
    {
        if (row >= mMaxBatchSize)
        {
            throw std::out_of_range("Batch::addKvCache: row is beyond max_batch_size");
        }
        if (mRows[static_cast<size_t>(row)] != nullptr)
        {
            throw LogicError("Batch::addKvCache: row is already occupied");
        }
    }

    // Point every layer group's index table at this row. The bind copies the current
    // indices in and keeps writing there, so the row needs no refresh of its own
    // afterwards -- and keeps doing so until the member leaves, which is why a step that
    // only advances its members has nothing to rebind.
    for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
    {
        // Beam 0 only; addKvCache rejects anything wider.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lg, _stagingRow(lg, row), mMaxBlocksPerSeq);
    }

    _seedRowWatermarks(row, kvCache.historyLength(), kvCache.capacity());

    // Joining is a stream change, so it carries the same cross-stream sync a direct move
    // would: work the member already enqueued has to retire before the Batch's stream
    // reaches anything that reads it.
    kvCache._setCudaStream(mCudaStream);

    // The arena lease goes with membership: a member is what gets stepped, so this is
    // exactly when the GPU cache is needed. After the stream change, because acquiring one
    // clears its residency on that stream. Sized from the same `constraints` the rows come
    // from, so a free row always has a free lease and this cannot fail.
    for (LifeCycleId lc{0}; lc < mNumLayerGroups; ++lc)
    {
        auto& arena = mManager->sparseArena(lc);
        if (arena.empty() || !kvCache.mSparseCache[lc].empty())
        {
            continue;
        }
        auto lease = arena.acquire(mCudaStream);
        TLLM_CHECK_WITH_INFO(lease.has_value(),
            "Batch::addKvCache: the sparse cache arena is full, which cannot happen while it is sized from the "
            "same `constraints` the Batch takes its rows from");
        kvCache.mSparseCache[lc] = std::move(*lease);
    }

    // Owning from here: the member cannot be destroyed under the Batch, so a caller may
    // keep only the Batch and get the reference back when it leaves.
    mRows[static_cast<size_t>(row)] = kvCache.shared_from_this();
    kvCache.mBatch = this;
    kvCache.mBatchRow = row;
    return row;
}

std::shared_ptr<KvCache> Batch::removeKvCache(int row)
{
    if (row < 0 || row >= mMaxBatchSize)
    {
        throw std::out_of_range("Batch::removeKvCache: row is beyond max_batch_size");
    }
    auto seated = mRows[static_cast<size_t>(row)];
    return seated ? removeKvCache(*seated) : nullptr;
}

std::shared_ptr<KvCache> Batch::removeKvCache(KvCache& kvCache)
{
    if (kvCache.mBatch != this)
    {
        throw LogicError("KvCache is not a member of this Batch");
    }
    waitPublished();
    int const row = kvCache.mBatchRow;
    // Held until the end: clearing the row below drops the Batch's reference, and the rest
    // of this runs against the member.
    auto leaving = std::move(mRows[static_cast<size_t>(row)]);
    // Membership ends first, so the unbinding below goes through the same setter a caller
    // would use. Nothing binds a seated member's buffer, including the Batch.
    kvCache.mBatch = nullptr;
    kvCache.mBatchRow = -1;
    for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
    {
        // Hands the indices back to the cache's own storage before the row is reused.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lg, nullptr, 0);
        auto* staged = _stagingRow(lg, row);
        std::fill(staged, staged + mMaxBlocksPerSeq, kBadPageIndex.value());
    }
    // Leaving returns the lease, which discards whatever the GPU cache held: a lease comes
    // back with every entry free. Anything worth keeping is written back by
    // flushSparseUnits() before this.
    for (auto& lease : kvCache.mSparseCache)
    {
        lease.release();
    }
    _seedRowWatermarks(row, 0, 0);
    mRows[static_cast<size_t>(row)] = nullptr;
    return leaving;
}

void Batch::reorder(std::vector<KvCache*> const& kvCaches)
{
    if (kvCaches.size() != static_cast<size_t>(numKvCaches()))
    {
        throw std::invalid_argument("Batch::reorder requires every current member exactly once");
    }
    std::vector<bool> seen(mRows.size(), false);
    for (auto const* cache : kvCaches)
    {
        if (cache == nullptr || cache->mBatch != this)
        {
            throw std::invalid_argument("Batch::reorder requires members of this Batch");
        }
        auto const row = static_cast<size_t>(cache->mBatchRow);
        if (seen[row])
        {
            throw std::invalid_argument("Batch::reorder requires unique members");
        }
        seen[row] = true;
    }

    waitPublished();
    for (size_t row = 0; row < kvCaches.size(); ++row)
    {
        int const oldRow = kvCaches[row]->mBatchRow;
        int const newRow = static_cast<int>(row);
        if (oldRow == newRow)
        {
            continue;
        }
        for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
        {
            auto* const oldIndices = _stagingRow(lg, oldRow);
            auto* const newIndices = _stagingRow(lg, newRow);
            std::swap_ranges(oldIndices, oldIndices + mMaxBlocksPerSeq, newIndices);
        }
        std::swap(mRows[static_cast<size_t>(oldRow)], mRows[row]);
        std::swap(mFlushedHistory[static_cast<size_t>(oldRow)], mFlushedHistory[row]);
        for (int const movedRow : {oldRow, newRow})
        {
            auto const& cache = mRows[static_cast<size_t>(movedRow)];
            if (cache != nullptr)
            {
                cache->mBatchRow = movedRow;
                for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
                {
                    // The row's contents have already moved. Rebind its sole writer
                    // without copying from the old row or ending cache membership.
                    cache->mBasePageIndices[kDefaultBeamIndex][lg]
                        = Span<int>{_stagingRow(lg, movedRow), mMaxBlocksPerSeq};
                }
            }
            _seedRowWatermarks(
                movedRow, cache == nullptr ? 0 : cache->historyLength(), cache == nullptr ? 0 : cache->capacity());
        }
    }
}

void Batch::_detach(KvCache& kvCache) noexcept
{
    if (kvCache.mBatch != this || kvCache.mBatchRow < 0)
    {
        return;
    }
    // A seated member holds a reference from mRows, so its destructor cannot run while it
    // is one. Reaching here means a row was cleared without unbinding the member, leaving a
    // row that publishes from an object about to stop existing.
    KVCM2_CHECK_FATAL_WITH_INFO(
        false, "A KvCache was destroyed while seated in a Batch, so its row outlived the cache it names");
}

std::vector<uint8_t> Batch::resize(
    std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths)
{
    if (capacities.size() != mRows.size() || historyLengths.size() != mRows.size())
    {
        throw std::invalid_argument("Batch::resize: one capacity and history length per row");
    }

    // Every row is checked before any is touched, so a bad request leaves the batch as it
    // was rather than part-way through.
    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr || !historyLengths[row].has_value())
        {
            continue;
        }
        int const newHist = *historyLengths[row];
        if (newHist < kvCache->historyLength())
        {
            throw std::invalid_argument("Batch::resize: history length cannot be decreased");
        }
        // Against the capacity that already existed, not the one being asked for. Capacity
        // this call grants makes room for tokens written after it, so history can only
        // cover tokens whose space was there while they were being written.
        if (newHist > kvCache->capacity())
        {
            throw std::invalid_argument("Batch::resize: history length cannot exceed the capacity the row already had");
        }
    }

    // Before the first member is touched, since each resize writes its staging row.
    waitPublished();

    std::vector<uint8_t> results(mRows.size(), 0);
    for (size_t row = 0; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr)
        {
            continue;
        }
        if (kvCache->resize(capacities[row], historyLengths[row]))
        {
            results[row] = 1;
            continue;
        }
        // The growth failed and rolled itself back, but the history advance is still both
        // legal -- it was checked against the capacity this row kept -- and free: the stale
        // range and the scratch range are each monotone in history, so raising it can only
        // release slots. Taking it anyway is what lets the units below it be written back
        // and their cells released, which is the pressure that made the growth fail.
        if (capacities[row].has_value() && historyLengths[row].has_value())
        {
            kvCache->resize(std::nullopt, historyLengths[row]);
        }
    }
    // Once, after every member has settled: a partial failure leaves the rolled-back state,
    // which is what the table should describe.
    publish();
    return results;
}

void Batch::publish()
{
    // A step's work-list counters start here, all of them at once. Per layer and per
    // direction they are independent, so one clear at the head of a step is enough -- and
    // clearing them here rather than before each producer keeps a non-kernel node from
    // standing between every resolve and whatever ran before it, which is what a launch
    // dependency cannot reach across.
    //
    // On the Batch's stream, like the staging below it. A fetch issued on another stream
    // therefore has to be ordered after this call -- but it already did, because it reads
    // the page tables and watermarks this same call stages.
    if (mCounterElements > 0)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            mCounters, 0, sizeof(int32_t) * mCounterElements, reinterpret_cast<cudaStream_t>(mCudaStream)));
    }

    // Refusing rather than waiting, because by this point a wait would be too late to help.
    // Everything that writes the staged rows has to be separated from the copy reading them:
    // the members' own mutations, which happened before this call, and the watermarks and
    // views staged just below. Waiting here would cover only the second, and would do it
    // while quietly hiding that the first had already raced.
    TLLM_CHECK_WITH_INFO(mPublished.isClosed(),
        "Batch::publish: the previous publish is still reading the staged rows; call waitPublished() before "
        "mutating any member or publishing again");

    // A SHARED life cycle gives every layer the same converted indices, which only holds
    // while no member substitutes scratch slots there. Prefill acquires those and decode
    // does not, so a batch that mixes them would have been given PER_LAYER; reaching here
    // means the phases were mixed after the mode was fixed.
    for (auto const& kvCache : mRows)
    {
        if (kvCache == nullptr)
        {
            continue;
        }
        for (LifeCycleId lcId{0}; lcId < mNumLayerGroups; ++lcId)
        {
            TLLM_CHECK_WITH_INFO(
                mIndexModes[lcId] != PageIndexMode::SHARED || kvCache->supportsIndexMode(PageIndexMode::SHARED, lcId),
                "Batch::publish: a member holds scratch slots in a SHARED life cycle; its layers no longer share "
                "one converted index table");
        }
    }

    // The members write their own page indices, but not their watermarks; collect those
    // now so the single transfer below carries a matched pair.
    _stageRowState();
    _stageScratch();

    auto const staging = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());
    auto* const upload = mUploadStream.handle();

    // The fill below overwrites what the previous step's copy read into the final tables.
    // Those run on different streams, so nothing but this event orders them.
    if (!mStagingRead.isClosed())
    {
        mStagingRead.waitInStream(reinterpret_cast<CudaStream>(upload));
        mStagingRead.close();
    }

    // One transfer for the whole table, rather than one per row whose locks moved. The
    // table is large but the copy is a single driver call on the upload stream, overlapping
    // compute; per-row transfers would trade that for a launch apiece, and the driver cost
    // of those dominates the bytes they save. Correctness does not depend on the choice --
    // a narrower refresh would be slower to decide, never wrong.
    cuCheck(cuMemcpyHtoDAsync(staging, reinterpret_cast<void*>(mStaging->address()), _totalBytes(), upload));
    if (mScratchStaging)
    {
        cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mScratchDevice.get()),
            reinterpret_cast<void*>(mScratchStaging->address()), sizeof(int32_t) * _scratchElements(), upload));
    }
    mPublished = mUploadStream.recordEvent();

    // The staging now holds this step's tables, so carrying them across is all that is
    // left on the Batch's stream -- one copy, because the two layouts match.
    mPublished.waitInStream(reinterpret_cast<CudaStream>(mCudaStream));
    cuCheck(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(mDevice.get()), staging, _deviceBytes(), mCudaStream));
    _convertIndices(mCudaStream);
    mStagingRead = CachedCudaEvent{reinterpret_cast<CudaStream>(mCudaStream)};
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

CoalescedPageTable Batch::coalescedPageTable(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::coalescedPageTable: unknown layer group");
    }
    CoalescedPageTable view;
    view.data = _stagingRow(layerGroupId, 0);
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

GpuCoalescedPageTable Batch::gpuCoalescedPageTable(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::gpuCoalescedPageTable: unknown layer group");
    }
    GpuCoalescedPageTable view;
    view.data = reinterpret_cast<CUdeviceptr>(mDevice.get())
        + sizeof(int32_t) * _rowElements() * static_cast<size_t>(layerGroupId.value());
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

CUdeviceptr Batch::sparseCacheViews(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::sparseCacheViews: unknown layer group");
    }
    return reinterpret_cast<CUdeviceptr>(mDevice.get()) + _viewsOffsetBytes()
        + sizeof(SparseCacheView) * static_cast<size_t>(layerGroupId.value()) * static_cast<size_t>(mMaxBatchSize);
}

WatermarkView Batch::watermarks() const
{
    auto const base = reinterpret_cast<CUdeviceptr>(mDevice.get());
    WatermarkView view;
    view.historyLength = base + sizeof(int32_t) * _historyOffset();
    view.capacity = base + sizeof(int32_t) * _capacityOffset();
    view.numRows = mMaxBatchSize;
    return view;
}

SparseHitMiss Batch::peekHitMissCounter() const
{
    if (!mHitMiss)
    {
        return {};
    }
    // A temporary stream, synced alone: the Batch's own stream carries the step, and
    // waiting on that would turn a statistics read into a pipeline stall.
    SparseHitMiss host;
    TemporaryCudaStream temp{{}};
    cuCheck(cuMemcpyDtoHAsync(&host, reinterpret_cast<CUdeviceptr>(mHitMiss.get()), sizeof(host), temp.get()));
    cuCheck(cuStreamSynchronize(temp.get()));
    return host;
}

void Batch::resetHitMissCounter()
{
    if (mHitMiss)
    {
        TLLM_CUDA_CHECK(
            cudaMemsetAsync(mHitMiss.get(), 0, sizeof(SparseHitMiss), reinterpret_cast<cudaStream_t>(mCudaStream)));
    }
}

size_t Batch::sparseArenaBytes() const
{
    size_t total = 0;
    for (LifeCycleId lc{0}; lc < mNumLayerGroups; ++lc)
    {
        total += mManager->sparseArena(lc).numBytes();
    }
    return total;
}

int Batch::numKvCaches() const noexcept
{
    return static_cast<int>(
        std::count_if(mRows.begin(), mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; }));
}

bool Batch::isContiguous() const noexcept
{
    // Nothing occupied past the first free row is the same thing as a prefix, and says so
    // without counting first.
    auto const firstFree = std::find(mRows.begin(), mRows.end(), nullptr);
    return std::none_of(firstFree, mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; });
}

int Batch::rowOf(KvCache const& kvCache) const
{
    return kvCache.mBatch == this ? kvCache.mBatchRow : -1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
