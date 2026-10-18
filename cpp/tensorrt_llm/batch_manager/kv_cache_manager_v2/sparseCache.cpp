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

#include "kv_cache_manager_v2/sparseCache.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/math.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{
namespace
{

constexpr size_t kAlignment = 256;

size_t alignedBytes(size_t bytes) noexcept
{
    return roundUp(bytes, kAlignment);
}

} // namespace

namespace
{

//! Per-lease metadata, in carve order. Arrays shared by every lease of a life cycle --
//! the buffer maps and the recency template -- live on the arena instead, so a resume
//! enqueues only the resets that differ per request.
struct LeaseLayout
{
    size_t ordinal = 0;
    size_t lru = 0;
    size_t view = 0;

    [[nodiscard]] size_t total() const noexcept
    {
        return ordinal + lru + view;
    }
};

LeaseLayout leaseLayout(SparseCacheInit const& shape)
{
    auto const entries = static_cast<size_t>(shape.numLayers) * shape.entriesPerLayer;
    LeaseLayout out;
    out.ordinal = alignedBytes(sizeof(int32_t) * entries);
    out.lru = alignedBytes(sizeof(int32_t) * entries);
    out.view = alignedBytes(sizeof(SparseCacheView));
    return out;
}

} // namespace

// ---------------------------------------------------------------------------
// SparseCellArena
// ---------------------------------------------------------------------------

SparseCellArena::SparseCellArena(SparseCacheInit const& shape, int maxLeases)
    : mShape(shape)
{
    if (shape.numLayers <= 0 || shape.numBuffers <= 0 || shape.entriesPerLayer <= 0 || shape.cellsPerBuffer <= 0)
    {
        throw std::invalid_argument("SparseCellArena requires a positive layer, buffer, entry and cell count");
    }
    if (shape.pools.empty() || maxLeases <= 0)
    {
        throw std::invalid_argument("SparseCellArena requires at least one pool and a positive lease count");
    }
    if (shape.bufferLayer.size() != static_cast<size_t>(shape.numBuffers))
    {
        throw std::invalid_argument("SparseCellArena requires one layer index per buffer");
    }
    if (shape.bufferSlotInPool.size() != static_cast<size_t>(shape.numBuffers))
    {
        throw std::invalid_argument("SparseCellArena requires one pool slot per buffer");
    }
    for (size_t i = 0; i < shape.bufferLayer.size(); ++i)
    {
        if (shape.bufferLayer[i] < 0 || shape.bufferLayer[i] >= shape.numLayers)
        {
            throw std::invalid_argument("SparseCellArena: a buffer's layer index is out of range");
        }
        if (i > 0 && shape.bufferLayer[i] < shape.bufferLayer[i - 1])
        {
            throw std::invalid_argument("SparseCellArena: buffers must be grouped by layer");
        }
    }
    for (auto const& [cellBytes, buffers] : shape.pools)
    {
        // {0, 0} is a pool index no sparse buffer of this life cycle uses: it holds its place
        // so that a storage pool index indexes this arena directly, and carries no cells. Any
        // other combination describes a pool that cannot be addressed.
        if ((cellBytes == 0) != (buffers == 0))
        {
            throw std::invalid_argument(
                "SparseCellArena: a pool needs both a cell size and buffers drawing on it, or neither");
        }
        if (buffers < 0)
        {
            throw std::invalid_argument("SparseCellArena: a pool cannot have a negative buffer count");
        }
    }

    auto const layout = leaseLayout(mShape);
    mMetadataBytes = layout.total();

    // One region per pool, each aligned so that a buffer's cell indices stay relative to its own
    // pool's base -- which is what get_sparse_cache_base_address reports for that buffer.
    std::vector<size_t> regionOffset;
    size_t cellBytes = 0;
    for (auto const& [bytes, buffers] : mShape.pools)
    {
        regionOffset.push_back(cellBytes);
        cellBytes += alignedBytes(static_cast<size_t>(maxLeases) * buffers * mShape.cellsPerBuffer * bytes);
    }

    auto const entries = static_cast<size_t>(mShape.numLayers) * mShape.entriesPerLayer;
    size_t const layerBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mShape.numBuffers));
    size_t const slotBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mShape.numBuffers));
    size_t const identityBytes = alignedBytes(sizeof(int32_t) * entries);
    size_t const sharedBytes = layerBytes + slotBytes + identityBytes;

    mNumBytes = cellBytes + sharedBytes + static_cast<size_t>(maxLeases) * mMetadataBytes;

    CUdeviceptr base = 0;
    cuCheck(cuMemAlloc(&base, mNumBytes));
    mMemory = CudaUniqPtr<>{reinterpret_cast<std::byte*>(base)};

    for (auto offset : regionOffset)
    {
        mPoolBase.push_back(base + offset);
    }
    auto* shared = reinterpret_cast<std::byte*>(base) + cellBytes;
    mSharedLayer = reinterpret_cast<int32_t*>(shared);
    mSharedSlotInPool = reinterpret_cast<int32_t*>(shared + layerBytes);
    mIdentity = reinterpret_cast<int32_t*>(shared + layerBytes + slotBytes);
    mMetadataBase = shared + sharedBytes;

    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mSharedLayer), mShape.bufferLayer.data(),
        sizeof(int32_t) * mShape.bufferLayer.size()));
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mSharedSlotInPool), mShape.bufferSlotInPool.data(),
        sizeof(int32_t) * mShape.bufferSlotInPool.size()));

    // The recency order every lease starts from: a per-layer identity permutation. Built once
    // here so a resume is a device-to-device copy rather than a host transfer.
    std::vector<int32_t> identity(entries);
    for (int32_t layer = 0; layer < mShape.numLayers; ++layer)
    {
        auto const begin = identity.begin() + static_cast<ptrdiff_t>(layer) * mShape.entriesPerLayer;
        std::iota(begin, begin + mShape.entriesPerLayer, 0);
    }
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mIdentity), identity.data(), sizeof(int32_t) * identity.size()));

    mFree.reserve(static_cast<size_t>(maxLeases));
    for (int i = maxLeases - 1; i >= 0; --i)
    {
        mFree.push_back(i);
    }
}

std::optional<SparseCacheLease> SparseCellArena::acquire(CUstream stream)
{
    if (empty() || mFree.empty())
    {
        return std::nullopt;
    }
    int const index = mFree.back();
    mFree.pop_back();

    auto const layout = leaseLayout(mShape);
    auto* bytes = mMetadataBase + static_cast<size_t>(index) * mMetadataBytes;
    size_t carved = 0;
    auto carve = [&](size_t size)
    {
        auto* ptr = reinterpret_cast<int32_t*>(bytes + carved);
        carved += size;
        return ptr;
    };

    SparseCacheLease lease;
    lease.mArena = this;
    lease.mIndex = index;

    auto& view = lease.mView;
    view.numLayers = mShape.numLayers;
    view.numBuffers = mShape.numBuffers;
    view.entriesPerLayer = mShape.entriesPerLayer;
    view.leaseIndex = index;
    view.bufferLayer = mSharedLayer;
    view.bufferSlotInPool = mSharedSlotInPool;
    view.residentOrdinal = carve(layout.ordinal);
    view.lruOrder = carve(layout.lru);
    lease.mDeviceView = reinterpret_cast<SparseCacheView*>(bytes + carved);

    auto const entries = static_cast<size_t>(view.numEntries());
    // Every entry starts free.
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(view.residentOrdinal),
        static_cast<unsigned int>(kBadBlockOrdinal.value()), entries, stream));
    cuCheck(cuMemcpyDtoDAsync(reinterpret_cast<CUdeviceptr>(view.lruOrder), reinterpret_cast<CUdeviceptr>(mIdentity),
        sizeof(int32_t) * entries, stream));
    // Last, so that it captures the carved pointers rather than a half-built view.
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(lease.mDeviceView), &view, sizeof(SparseCacheView)));
    return lease;
}

void SparseCellArena::_release(int index) noexcept
{
    if (index >= 0)
    {
        mFree.push_back(index);
    }
}

// ---------------------------------------------------------------------------
// SparseCacheLease
// ---------------------------------------------------------------------------

SparseCacheLease::~SparseCacheLease()
{
    release();
}

SparseCacheLease::SparseCacheLease(SparseCacheLease&& other) noexcept
    : mArena(other.mArena)
    , mIndex(other.mIndex)
    , mView(other.mView)
    , mDeviceView(other.mDeviceView)
{
    other.mArena = nullptr;
    other.mIndex = -1;
}

SparseCacheLease& SparseCacheLease::operator=(SparseCacheLease&& other) noexcept
{
    if (this != &other)
    {
        release();
        mArena = other.mArena;
        mIndex = other.mIndex;
        mView = other.mView;
        mDeviceView = other.mDeviceView;
        other.mArena = nullptr;
        other.mIndex = -1;
    }
    return *this;
}

void SparseCacheLease::release() noexcept
{
    if (mArena != nullptr)
    {
        mArena->_release(mIndex);
        mArena = nullptr;
        mIndex = -1;
        mView = SparseCacheView{};
        mDeviceView = nullptr;
    }
}

std::vector<int32_t> SparseCacheLease::readResidentOrdinals() const
{
    if (empty())
    {
        return {};
    }
    std::vector<int32_t> out(mView.numEntries());
    cuCheck(
        cuMemcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(mView.residentOrdinal), sizeof(int32_t) * out.size()));
    return out;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
