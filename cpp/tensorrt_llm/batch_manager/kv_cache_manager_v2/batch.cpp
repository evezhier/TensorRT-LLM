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
#include "kv_cache_manager_v2/indexConversionKernels.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

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

Batch::Batch(KvCacheManager& manager, int maxBatchSize, int maxBlocksPerSeq, std::optional<ExternalPageTable> pageTable,
    std::optional<PageIndexMode> mode, std::vector<IndexTableRequest> indexTables)
    : mManager(&manager)
    , mPageTable(pageTable)
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

    if (mPageTable)
    {
        if (mPageTable->host == nullptr)
        {
            throw std::invalid_argument("Batch: a borrowed page table needs a host address");
        }
        if (mPageTable->rowStride < mMaxBlocksPerSeq
            || mPageTable->layerGroupStride < mPageTable->rowStride * mMaxBatchSize)
        {
            throw std::invalid_argument("Batch: borrowed page table strides are too small for its rows");
        }
    }

    // Page tables and watermarks are published together, so a kernel cannot read one step's
    // table beside another step's lengths -- an out-of-range read rather than a stale
    // answer. When the page table is borrowed they travel as separate transfers on one
    // stream, which orders them just as well.
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
    CUdeviceptr device = 0;
    cuCheck(cuMemAlloc(&device, _deviceBytes()));
    mDevice = CudaUniqPtr{reinterpret_cast<std::byte*>(device)};
    CUdeviceptr deviceStaging = 0;
    cuCheck(cuMemAlloc(&deviceStaging, _deviceBytes()));
    mDeviceStaging = CudaUniqPtr{reinterpret_cast<std::byte*>(deviceStaging)};

    for (auto const base : {device, deviceStaging})
    {
        if (_deviceBias() > 0)
        {
            // The borrowed rows are the lender's until a member takes one, so there is
            // nothing to copy in yet; the table still has to read as absent until the
            // first publish.
            cuCheck(cuMemsetD32(base, static_cast<unsigned int>(kBadPageIndex.value()),
                _rowElements() * static_cast<size_t>(mNumLayerGroups.value())));
        }
        cuCheck(cuMemcpyHtoD(base + _deviceBias(), staging, _totalBytes()));
    }

    _allocateFetchScratch();

    // After the modes: a request's offsets depend on where its life cycle puts the base.
    _buildIndexTables(indexTables);
    _allocateIndexTables();
}

void Batch::_allocateFetchScratch()
{
    mScratch = TypedVec<LifeCycleId, SparseFetchScratch>(mNumLayerGroups, SparseFetchScratch{});

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
        total += rows * (3 * static_cast<size_t>(width) + 1 + static_cast<size_t>(scratch.maxInFlightUnits));
    }
    if (total == 0)
    {
        return;
    }

    CUdeviceptr base = 0;
    cuCheck(cuMemAlloc(&base, sizeof(int32_t) * total));
    mFetchScratch = CudaUniqPtr{reinterpret_cast<std::byte*>(base)};

    auto* cursor = reinterpret_cast<int32_t*>(base);
    auto carve = [&cursor](size_t count)
    {
        auto* ptr = cursor;
        cursor += count;
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
        scratch.missOrdinal = carve(rows * width);
        scratch.missSlot = carve(rows * width);
        scratch.missCount = carve(rows);
        scratch.inFlightSlot = carve(rows * static_cast<size_t>(scratch.maxInFlightUnits));
    }
}

SparseFetchScratch const& Batch::sparseFetchScratch(LayerGroupId layerGroupId) const
{
    LifeCycleId const lcId{layerGroupId.value()};
    if (lcId >= mScratch.size())
    {
        throw std::out_of_range("Batch::sparseFetchScratch: no such layer group");
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
        CUdeviceptr scratch = 0;
        cuCheck(cuMemAlloc(&scratch, sizeof(int32_t) * elements));
        mScratchDevice = CudaUniqPtr{reinterpret_cast<std::byte*>(scratch)};
        cuCheck(cuMemcpyHtoD(scratch, host, sizeof(int32_t) * elements));
    }

    auto const rowBlocks = static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
    size_t elements = 0;
    for (auto& table : mIndexTables)
    {
        table.elementOffset = elements;
        elements += table.concatOffsets.size() * rowBlocks;
        mMaxTableEntries = std::max(mMaxTableEntries, static_cast<int>(table.concatOffsets.size()));
    }

    CUdeviceptr memory = 0;
    cuCheck(cuMemAlloc(&memory, sizeof(int32_t) * elements));
    mIndexTableMemory = CudaUniqPtr{reinterpret_cast<std::byte*>(memory)};
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

    CUdeviceptr descs = 0;
    cuCheck(cuMemAlloc(&descs, descBytes + offsetsBytes));
    mIndexTableDescs = CudaUniqPtr{reinterpret_cast<std::byte*>(descs)};

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
        auto const* kvCache = mRows[row];
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

int64_t Batch::_pageTableRowStride() const noexcept
{
    return mPageTable ? mPageTable->rowStride : static_cast<int64_t>(mMaxBlocksPerSeq);
}

int64_t Batch::_pageTableGroupStride() const noexcept
{
    return mPageTable ? mPageTable->layerGroupStride : static_cast<int64_t>(_rowElements());
}

size_t Batch::_deviceBias() const noexcept
{
    if (!mPageTable)
    {
        return 0;
    }
    // Aligned for the views, so adding this to a staging offset lands on the same field of
    // the device block rather than one short of it.
    auto const bytes = sizeof(int32_t) * _rowElements() * static_cast<size_t>(mNumLayerGroups.value());
    auto const align = alignof(SparseCacheView);
    return (bytes + align - 1) / align * align;
}

size_t Batch::_deviceBytes() const noexcept
{
    return _deviceBias() + _totalBytes();
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
    // A borrowed table occupies none of the staged block, so the watermarks start it.
    return mPageTable ? 0 : _rowElements() * static_cast<size_t>(mNumLayerGroups.value());
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
        auto const* kvCache = mRows[row];
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

int32_t* Batch::_stagingRow(LayerGroupId layerGroupId, int row) const
{
    auto* base = mPageTable ? mPageTable->host : reinterpret_cast<int32_t*>(mStaging->address());
    auto const offset
        = _pageTableGroupStride() * layerGroupId.value() + _pageTableRowStride() * static_cast<int64_t>(row);
    return base + offset;
}

int Batch::addKvCache(KvCache& kvCache, int requestedRow)
{
    if (kvCache.mBatch != nullptr)
    {
        throw LogicError("A KvCache belongs to at most one Batch");
    }
    if (mPageTable && requestedRow < 0)
    {
        // Choosing one would address whichever request the lender seated there.
        throw LogicError("A Batch borrowing a page table must be told which row a member occupies");
    }
    if (kvCache.beamWidth() != BeamIndex{1})
    {
        // Only beam 0 is mirrored below, so a wider request would publish one beam's pages
        // and have every beam read them. Refusing beats a wrong answer nothing would notice.
        throw LogicError("Batch does not support beam search yet; a member must have beam width 1");
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
        kvCache._bindBasePageIndexBuf(kDefaultBeamIndex, lg, _stagingRow(lg, row), mMaxBlocksPerSeq);
    }

    mRows[static_cast<size_t>(row)] = &kvCache;
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
        kvCache._bindBasePageIndexBuf(kDefaultBeamIndex, lg, nullptr, 0);
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
    for (auto const* kvCache : mRows)
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

    // Whole-table for now. Refreshing only the rows whose locks moved is an optimisation
    // the design allows for: it degrades into a slower step, never a wrong answer.
    if (mPageTable)
    {
        // One transfer per layer group rather than one for all of them: the lender's rows
        // are further apart than they are wide, and its groups further apart still, so no
        // single stride describes the source. Rows no member occupies are carried along
        // because skipping them would cost a transfer each.
        for (LayerGroupId lg{0}; lg < mNumLayerGroups; ++lg)
        {
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = _stagingRow(lg, 0);
            copy.srcPitch = sizeof(int32_t) * static_cast<size_t>(_pageTableRowStride());
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = staging + sizeof(int32_t) * _rowElements() * static_cast<size_t>(lg.value());
            copy.dstPitch = sizeof(int32_t) * static_cast<size_t>(mMaxBlocksPerSeq);
            copy.WidthInBytes = sizeof(int32_t) * static_cast<size_t>(mMaxBlocksPerSeq);
            copy.Height = static_cast<size_t>(mMaxBatchSize);
            cuCheck(cuMemcpy2DAsync(&copy, upload));
        }
    }
    cuCheck(cuMemcpyHtoDAsync(
        staging + _deviceBias(), reinterpret_cast<void*>(mStaging->address()), _totalBytes(), upload));
    if (mScratchStaging)
    {
        cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mScratchDevice.get()),
            reinterpret_cast<void*>(mScratchStaging->address()), sizeof(int32_t) * _scratchElements(), upload));
    }
    mPublished = mUploadStream.recordEvent();

    // The staging now holds this step's tables, so carrying them across is all that is
    // left on the caller's stream -- one copy, because the two layouts match.
    mPublished.waitInStream(reinterpret_cast<CudaStream>(stream));
    cuCheck(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(mDevice.get()), staging, _deviceBytes(), stream));
    _convertIndices(stream);
    mStagingRead = CachedCudaEvent{reinterpret_cast<CudaStream>(stream)};
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

CUdeviceptr Batch::sparseCacheViews(LayerGroupId layerGroupId) const
{
    if (layerGroupId.value() < 0 || layerGroupId >= mNumLayerGroups)
    {
        throw std::out_of_range("Batch::sparseCacheViews: unknown layer group");
    }
    return reinterpret_cast<CUdeviceptr>(mDevice.get()) + _deviceBias() + _viewsOffsetBytes()
        + sizeof(SparseCacheView) * static_cast<size_t>(layerGroupId.value()) * static_cast<size_t>(mMaxBatchSize);
}

WatermarkView Batch::watermarks() const
{
    auto const base = reinterpret_cast<CUdeviceptr>(mDevice.get()) + _deviceBias();
    WatermarkView view;
    view.historyLength = base + sizeof(int32_t) * _historyOffset();
    view.capacity = base + sizeof(int32_t) * _capacityOffset();
    view.numRows = mMaxBatchSize;
    return view;
}

int Batch::rowOf(KvCache const& kvCache) const
{
    return kvCache.mBatch == this ? kvCache.mBatchRow : -1;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
