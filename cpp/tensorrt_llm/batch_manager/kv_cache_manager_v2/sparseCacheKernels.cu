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

#include "sparseCacheKernels.h"

#include "exceptions.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

constexpr int kThreads = 256;
constexpr int32_t kEmptyKey = -1;
//! Evaluated on the host so device code never calls StrongIndex::value().
constexpr int32_t kFreeOrdinal = kBadBlockOrdinal.value();

//! Recency classes, ordered: a hit outranks a unit fetched this step, which outranks an
//! entry the step did not touch. This is what makes a repeatedly selected unit survive a
//! one-off selection.
constexpr uint8_t kClassHit = 0;
constexpr uint8_t kClassFetched = 1;
constexpr uint8_t kClassUntouched = 2;

using BlockScan = cub::BlockScan<int32_t, kThreads>;

__host__ __device__ inline size_t alignUp16(size_t bytes) noexcept
{
    return (bytes + 15U) & ~static_cast<size_t>(15U);
}

//! Byte offsets of the shared-memory regions one CTA carves. Computed identically on the
//! host, which sizes the launch, and in the kernel, which carves it. Every region is
//! 16-byte aligned: cls is a byte array, so an unaligned int32 region after it faults.
struct SmemLayout
{
    size_t keys = 0;
    size_t vals = 0;
    size_t cls = 0;
    size_t scratch = 0;
    size_t order = 0;
    size_t missIdx = 0;
    size_t total = 0;
};

__host__ __device__ inline SmemLayout smemLayout(int32_t tableSize, int32_t entries, int32_t width) noexcept
{
    SmemLayout layout;
    size_t offset = 0;
    layout.keys = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(tableSize));
    layout.vals = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(tableSize));
    layout.cls = offset;
    offset = alignUp16(offset + sizeof(uint8_t) * static_cast<size_t>(entries));
    layout.scratch = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(entries));
    layout.order = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(entries));
    layout.missIdx = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(width));
    layout.total = offset;
    return layout;
}

[[nodiscard]] constexpr int32_t nextPow2(int32_t value) noexcept
{
    int32_t result = 1;
    while (result < value)
    {
        result <<= 1;
    }
    return result;
}

//! Fibonacci hashing. Unit ordinals are small and dense, and identity hashing turns a dense
//! run into one long probe chain.
__device__ inline uint32_t slotOf(int32_t key, uint32_t mask) noexcept
{
    return (static_cast<uint32_t>(key) * 0x9E3779B1U) & mask;
}

//! Exclusive sum of a per-position flag over [0, n), single CTA, result in `flags`.
//! Returns the total. `flags` is shared memory; the scan is in place.
__device__ int32_t blockExclusiveSum(int32_t* flags, int32_t n, typename BlockScan::TempStorage& temp)
{
    __shared__ int32_t carry;
    if (threadIdx.x == 0)
    {
        carry = 0;
    }
    __syncthreads();

    for (int32_t base = 0; base < n; base += kThreads)
    {
        int32_t const idx = base + static_cast<int32_t>(threadIdx.x);
        int32_t const value = idx < n ? flags[idx] : 0;
        int32_t scanned = 0;
        int32_t aggregate = 0;
        BlockScan(temp).ExclusiveSum(value, scanned, aggregate);
        if (idx < n)
        {
            flags[idx] = carry + scanned;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            carry += aggregate;
        }
        __syncthreads();
    }
    return carry;
}

//! One CTA per layer.
//!
//! Shared memory is carved by hand because the probe index is sized at launch; see
//! SmemLayout for the regions and their alignment.
__global__ __launch_bounds__(kThreads) void sparseResolveKernel(SparseResolveArgs args, int32_t tableSize)
{
    extern __shared__ unsigned char smem[];
    int32_t const layer = static_cast<int32_t>(blockIdx.x);
    int32_t const entries = args.cache.entriesPerLayer;
    int32_t const width = args.selectionWidth;
    uint32_t const mask = static_cast<uint32_t>(tableSize) - 1U;

    SmemLayout const layout = smemLayout(tableSize, entries, width);
    auto* keys = reinterpret_cast<int32_t*>(smem + layout.keys);
    auto* vals = reinterpret_cast<int32_t*>(smem + layout.vals);
    auto* cls = reinterpret_cast<uint8_t*>(smem + layout.cls);
    auto* scratch = reinterpret_cast<int32_t*>(smem + layout.scratch);
    auto* order = reinterpret_cast<int32_t*>(smem + layout.order);
    auto* missIdx = reinterpret_cast<int32_t*>(smem + layout.missIdx);

    __shared__ typename BlockScan::TempStorage temp;
    __shared__ int32_t numMiss;

    int32_t* residentOrdinal = args.cache.residentOrdinal + static_cast<ptrdiff_t>(layer) * entries;
    int32_t* lruOrder = args.cache.lruOrder + static_cast<ptrdiff_t>(layer) * entries;
    int32_t const* selected = args.selected + static_cast<ptrdiff_t>(layer) * width;
    int32_t* outSlot = args.outSlot + static_cast<ptrdiff_t>(layer) * width;
    int32_t* missOrdinal = args.missOrdinal + static_cast<ptrdiff_t>(layer) * width;
    int32_t* missSlot = args.missSlot + static_cast<ptrdiff_t>(layer) * width;

    if (threadIdx.x == 0)
    {
        numMiss = 0;
    }
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < tableSize; i += kThreads)
    {
        keys[i] = kEmptyKey;
    }
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < entries; i += kThreads)
    {
        cls[i] = kClassUntouched;
    }
    __syncthreads();

    // Build the probe index over the resident set. Ordinals are unique within a layer, so
    // threads contend only on table-position collisions.
    for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
    {
        int32_t const ord = residentOrdinal[s];
        if (ord == kFreeOrdinal)
        {
            continue;
        }
        uint32_t pos = slotOf(ord, mask);
        while (atomicCAS(&keys[pos], kEmptyKey, ord) != kEmptyKey)
        {
            pos = (pos + 1U) & mask;
        }
        vals[pos] = s;
    }
    __syncthreads();

    // Probe once per selection. A hit resolves immediately; a miss joins the list the copy
    // stage will read, so copies can be issued before any recency bookkeeping runs.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < width; i += kThreads)
    {
        int32_t const ord = selected[i];
        if (ord < 0)
        {
            outSlot[i] = kNoSparseSlot;
            continue;
        }
        uint32_t pos = slotOf(ord, mask);
        int32_t key = keys[pos];
        while (key != kEmptyKey && key != ord)
        {
            pos = (pos + 1U) & mask;
            key = keys[pos];
        }
        if (key == ord)
        {
            int32_t const slot = vals[pos];
            outSlot[i] = slot;
            cls[slot] = kClassHit;
        }
        else
        {
            int32_t const j = atomicAdd(&numMiss, 1);
            missOrdinal[j] = ord;
            missIdx[j] = i;
        }
    }
    __syncthreads();

    // Entries that may be given away: free ones, and resident ones this step did not select
    // that are not holding in-flight data.
    for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
    {
        int32_t const slot = lruOrder[p];
        int32_t const ord = residentOrdinal[slot];
        bool const free = ord == kFreeOrdinal;
        bool const pinned = !free && ord >= args.historyUnits;
        scratch[p] = (free || (cls[slot] == kClassUntouched && !pinned)) ? 1 : 0;
    }
    __syncthreads();

    // Victims come from the tail, least recently used first, so rank from the tail.
    int32_t const totalEvictable = blockExclusiveSum(scratch, entries, temp);
    if (totalEvictable < numMiss)
    {
        if (threadIdx.x == 0)
        {
            atomicExch(args.status, static_cast<int32_t>(SparseResolveStatus::kOutOfEntries));
        }
        return;
    }
    __syncthreads();

    for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
    {
        int32_t const slot = lruOrder[p];
        int32_t const ord = residentOrdinal[slot];
        bool const free = ord == kFreeOrdinal;
        bool const pinned = !free && ord >= args.historyUnits;
        bool const evictable = free || (cls[slot] == kClassUntouched && !pinned);
        if (!evictable)
        {
            continue;
        }
        // scratch[p] is the count of evictable entries before p, so this is its rank from
        // the tail. The numMiss least recently used take the misses, in selection order.
        int32_t const rankFromTail = totalEvictable - 1 - scratch[p];
        if (rankFromTail < numMiss)
        {
            int32_t const j = numMiss - 1 - rankFromTail;
            residentOrdinal[slot] = missOrdinal[j];
            missSlot[j] = slot;
            outSlot[missIdx[j]] = slot;
            cls[slot] = kClassFetched;
        }
    }
    __syncthreads();

    // Rebuild recency: hits, then units fetched this step, then everything untouched, each
    // group keeping its previous relative order.
    int32_t base[3];
    for (uint8_t c = 0; c < 3; ++c)
    {
        for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
        {
            scratch[p] = cls[lruOrder[p]] == c ? 1 : 0;
        }
        __syncthreads();
        int32_t const total = blockExclusiveSum(scratch, entries, temp);
        base[c] = total;
        for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
        {
            if (cls[lruOrder[p]] == c)
            {
                // Offset by the preceding classes once their totals are known, below.
                order[p] = scratch[p];
            }
        }
        __syncthreads();
    }

    int32_t const hitBase = 0;
    int32_t const fetchedBase = base[0];
    int32_t const untouchedBase = base[0] + base[1];
    for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
    {
        int32_t const slot = lruOrder[p];
        uint8_t const c = cls[slot];
        int32_t const offset = c == kClassHit ? hitBase : (c == kClassFetched ? fetchedBase : untouchedBase);
        scratch[offset + order[p]] = slot;
    }
    __syncthreads();

    for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
    {
        lruOrder[p] = scratch[p];
    }
    if (threadIdx.x == 0)
    {
        args.missCount[layer] = numMiss;
    }
}

} // namespace

size_t sparseResolveSharedBytes(int32_t entriesPerLayer, int32_t selectionWidth)
{
    return smemLayout(nextPow2(2 * entriesPerLayer), entriesPerLayer, selectionWidth).total;
}

int32_t sparseResolveMaxEntries(int32_t selectionWidth, CUdevice device)
{
    int limit = 0;
    cuCheck(cuDeviceGetAttribute(&limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
    int32_t best = 0;
    for (int32_t entries = 1; entries <= (1 << 20); entries <<= 1)
    {
        if (sparseResolveSharedBytes(entries, selectionWidth) > static_cast<size_t>(limit))
        {
            break;
        }
        best = entries;
    }
    return best;
}

void launchSparseResolve(SparseResolveArgs const& args, CUstream stream)
{
    if (args.cache.numLayers <= 0 || args.cache.entriesPerLayer <= 0 || args.selectionWidth <= 0)
    {
        throw std::invalid_argument("launchSparseResolve: layers, entries and selection width must be positive");
    }
    if (args.selected == nullptr || args.outSlot == nullptr || args.missOrdinal == nullptr || args.missSlot == nullptr
        || args.missCount == nullptr || args.status == nullptr)
    {
        throw std::invalid_argument("launchSparseResolve: every device array must be provided");
    }

    int32_t const tableSize = nextPow2(2 * args.cache.entriesPerLayer);
    size_t const shared = sparseResolveSharedBytes(args.cache.entriesPerLayer, args.selectionWidth);

    CUdevice device = 0;
    cuCheck(cuCtxGetDevice(&device));
    int limit = 0;
    cuCheck(cuDeviceGetAttribute(&limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
    if (shared > static_cast<size_t>(limit))
    {
        throw std::invalid_argument("launchSparseResolve: the cache needs " + std::to_string(shared)
            + " bytes of shared memory, above this device's limit of " + std::to_string(limit)
            + "; reduce max_cached_units or raise the selection unit size");
    }

    if (shared > 48U * 1024U)
    {
        TLLM_CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<void const*>(&sparseResolveKernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
    }
    sparseResolveKernel<<<args.cache.numLayers, kThreads, shared, reinterpret_cast<cudaStream_t>(stream)>>>(
        args, tableSize);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
