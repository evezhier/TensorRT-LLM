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

SparseCacheState::SparseCacheState(
    int32_t numLayers, int32_t entriesPerLayer, std::vector<int32_t> const& cachePageSlots, CUstream stream)
{
    if (numLayers <= 0 || entriesPerLayer <= 0)
    {
        throw std::invalid_argument("SparseCacheState requires a positive layer and entry count");
    }
    if (cachePageSlots.empty())
    {
        throw std::invalid_argument("SparseCacheState requires at least one cache page");
    }

    mView.numLayers = numLayers;
    mView.entriesPerLayer = entriesPerLayer;
    mView.numPages = static_cast<int32_t>(cachePageSlots.size());

    size_t const ordinalBytes = alignedBytes(sizeof(int32_t) * mView.numEntries());
    size_t const lruBytes = alignedBytes(sizeof(int32_t) * mView.numEntries());
    size_t const pageBytes = alignedBytes(sizeof(int32_t) * cachePageSlots.size());

    CUdeviceptr base = 0;
    cuCheck(cuMemAlloc(&base, ordinalBytes + lruBytes + pageBytes));
    mMemory = CudaUniqPtr{reinterpret_cast<std::byte*>(base)};

    auto* bytes = reinterpret_cast<std::byte*>(base);
    mView.residentOrdinal = reinterpret_cast<int32_t*>(bytes);
    mView.lruOrder = reinterpret_cast<int32_t*>(bytes + ordinalBytes);
    mView.cachePages = reinterpret_cast<int32_t*>(bytes + ordinalBytes + lruBytes);
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
