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

SparseCacheState::SparseCacheState(SparseCacheInit const& init, CUstream stream)
{
    int32_t const numLayers = init.numLayers;
    int32_t const entriesPerLayer = init.entriesPerLayer;
    auto const& cachePageSlots = init.cachePageSlots;
    if (numLayers <= 0 || entriesPerLayer <= 0)
    {
        throw std::invalid_argument("SparseCacheState requires a positive layer and entry count");
    }
    if (cachePageSlots.empty())
    {
        throw std::invalid_argument("SparseCacheState requires at least one cache page");
    }
    if (init.hostBufferOffset.size() != static_cast<size_t>(numLayers))
    {
        throw std::invalid_argument("SparseCacheState requires one cold-page offset per buffer");
    }

    mView.numLayers = numLayers;
    mView.entriesPerLayer = entriesPerLayer;
    mView.numPages = static_cast<int32_t>(cachePageSlots.size());
    mView.maxFlushUnits = std::max(init.maxFlushUnits, 1);
    mView.maxFlushBlocks = std::max(init.maxFlushBlocks, 1);

    auto const flushEntries = static_cast<size_t>(numLayers) * mView.maxFlushUnits;
    size_t const ordinalBytes = alignedBytes(sizeof(int32_t) * mView.numEntries());
    size_t const lruBytes = alignedBytes(sizeof(int32_t) * mView.numEntries());
    size_t const pageBytes = alignedBytes(sizeof(int32_t) * cachePageSlots.size());
    size_t const flushListBytes = alignedBytes(sizeof(int32_t) * flushEntries);
    size_t const countBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(numLayers));
    size_t const rangeBytes = alignedBytes(sizeof(int32_t) * 2);
    size_t const sliceBytes = alignedBytes(sizeof(int32_t) * static_cast<size_t>(mView.maxFlushBlocks));
    size_t const offsetBytes = alignedBytes(sizeof(uint32_t) * static_cast<size_t>(numLayers));

    CUdeviceptr base = 0;
    cuCheck(cuMemAlloc(&base,
        ordinalBytes + lruBytes + pageBytes + 2 * flushListBytes + countBytes + rangeBytes + sliceBytes + offsetBytes));
    mMemory = CudaUniqPtr{reinterpret_cast<std::byte*>(base)};

    auto* bytes = reinterpret_cast<std::byte*>(base);
    size_t carved = 0;
    auto carve = [&](size_t size)
    {
        auto* ptr = reinterpret_cast<int32_t*>(bytes + carved);
        carved += size;
        return ptr;
    };
    mView.residentOrdinal = carve(ordinalBytes);
    mView.lruOrder = carve(lruBytes);
    mView.cachePages = carve(pageBytes);
    mView.flushOrdinal = carve(flushListBytes);
    mView.flushSlot = carve(flushListBytes);
    mView.flushCount = carve(countBytes);
    mView.flushRange = carve(rangeBytes);
    mView.flushPageSlice = carve(sliceBytes);
    mView.hostBufferOffset = reinterpret_cast<uint32_t*>(carve(offsetBytes));
    // -1 so that "no write-back has run" is distinguishable from "one ran and found
    // nothing to move", which is what lets a test tell the two apart.
    cuCheck(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(mView.flushCount), 0xFFFFFFFFU, static_cast<size_t>(numLayers), stream));
    // Synchronous: both sources are the caller's and may go away on return.
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mView.hostBufferOffset), init.hostBufferOffset.data(),
        sizeof(uint32_t) * init.hostBufferOffset.size()));
    // Synchronous: cachePageSlots belongs to the caller and may go away on return.
    cuCheck(cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(mView.cachePages), cachePageSlots.data(),
        sizeof(int32_t) * cachePageSlots.size()));

    // Every entry starts free. cuMemsetD32Async covers the ordinal sentinel; the recency order
    // is a per-layer identity permutation, so it is built on the host and copied rather than
    // written by a kernel this package would otherwise not need.
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(mView.residentOrdinal),
        static_cast<unsigned int>(kBadBlockOrdinal.value()), mView.numEntries(), stream));

    std::vector<int32_t> identity(mView.numEntries());
    for (int32_t layer = 0; layer < numLayers; ++layer)
    {
        auto const begin = identity.begin() + static_cast<ptrdiff_t>(layer) * entriesPerLayer;
        std::iota(begin, begin + entriesPerLayer, 0);
    }
    // Synchronous: `identity` is a stack temporary, so the copy must complete before return.
    cuCheck(cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(mView.lruOrder), identity.data(), sizeof(int32_t) * identity.size()));
}

std::vector<int32_t> SparseCacheState::readFlushCounts() const
{
    if (empty())
    {
        return {};
    }
    std::vector<int32_t> out(static_cast<size_t>(mView.numLayers));
    cuCheck(cuMemcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(mView.flushCount), sizeof(int32_t) * out.size()));
    return out;
}

std::vector<int32_t> SparseCacheState::readCachePages() const
{
    if (empty())
    {
        return {};
    }
    std::vector<int32_t> out(static_cast<size_t>(mView.numPages));
    cuCheck(cuMemcpyDtoH(out.data(), reinterpret_cast<CUdeviceptr>(mView.cachePages), sizeof(int32_t) * out.size()));
    return out;
}

std::vector<int32_t> SparseCacheState::readResidentOrdinals() const
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
