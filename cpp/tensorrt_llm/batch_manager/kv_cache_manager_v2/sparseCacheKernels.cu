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

#include <cassert>

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Enforce a precondition that correct use cannot violate.
//!
//! There is no reporting channel: trapping destroys the context, so nothing survives to
//! carry a status home, and the host checks that would catch these cases legitimately run
//! before the launch. A debug build names the condition; a release build just halts, which
//! still beats proceeding with indices that would be read as real.
#define KVCM2_SPARSE_REQUIRE(cond)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            assert(!"KVCM2 sparse invariant violated: " #cond);                                                        \
            __trap();                                                                                                  \
        }                                                                                                              \
    } while (0)

constexpr int kThreads = 256;
constexpr int32_t kEmptyKey = -1;
//! kBadPageIndex as a plain int32, since device code cannot use the strong typedef.
constexpr int32_t kBadPageIndexValue = kBadPageIndex.value();
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
    size_t missOrd = 0;
    size_t total = 0;
};

//! @param required Length of the virtual list the probe walks: in-flight units first,
//! then the selection.
__host__ __device__ inline SmemLayout smemLayout(int32_t tableSize, int32_t entries, int32_t required) noexcept
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
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(required));
    layout.missOrd = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(required));
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
    // One CTA per row: the call resolves a single layer, whose metadata row every sparse
    // buffer of that layer shares.
    int32_t const layer = args.layerInGroup;
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    SparseCacheView const view = args.views[row];
    if (view.residentOrdinal == nullptr)
    {
        // A free row, or a member that has suspended and given its reservation back. Its
        // outputs keep whatever the caller initialised them to.
        return;
    }
    int32_t const entries = view.entriesPerLayer;
    int32_t const width = args.selectionWidth;
    uint32_t const mask = static_cast<uint32_t>(tableSize) - 1U;

    // Both watermarks are read from device memory: they move every step, so a captured
    // graph must see the current value.
    int32_t const historyUnits = args.historyTokens[row] / args.unitTokens;
    int32_t const inFlight = args.capacityTokens[row] / args.unitTokens - historyUnits;
    // resize() rejects a gap wider than the cache can pin, so this cannot hold.
    KVCM2_SPARSE_REQUIRE(inFlight >= 0 && inFlight <= args.maxInFlightUnits);
    // The probe walks one virtual list: the in-flight units, which must be resident before
    // the selection takes any victim, then the selection itself.
    int32_t const required = inFlight + width;

    SmemLayout const layout = smemLayout(tableSize, entries, args.maxInFlightUnits + width);
    auto* keys = reinterpret_cast<int32_t*>(smem + layout.keys);
    auto* vals = reinterpret_cast<int32_t*>(smem + layout.vals);
    auto* cls = reinterpret_cast<uint8_t*>(smem + layout.cls);
    auto* scratch = reinterpret_cast<int32_t*>(smem + layout.scratch);
    auto* order = reinterpret_cast<int32_t*>(smem + layout.order);
    auto* missIdx = reinterpret_cast<int32_t*>(smem + layout.missIdx);
    auto* missOrd = reinterpret_cast<int32_t*>(smem + layout.missOrd);

    __shared__ typename BlockScan::TempStorage temp;
    __shared__ int32_t numMiss;
    __shared__ int32_t numCopy;

    // The view's own arrays belong to this row's allocation, so only the layer indexes them.
    int32_t* residentOrdinal = view.residentOrdinal + static_cast<ptrdiff_t>(layer) * entries;
    int32_t* lruOrder = view.lruOrder + static_cast<ptrdiff_t>(layer) * entries;
    // The batch-wide arrays are indexed by row alone: the call is one layer, and its
    // buffers share the single result.
    auto const selectionBase = static_cast<ptrdiff_t>(row) * width;
    auto const inFlightBase = static_cast<ptrdiff_t>(row) * args.maxInFlightUnits;
    int32_t const* selected = args.selected + selectionBase;
    int32_t* outSlot = args.outSlot + selectionBase;
    int32_t* missOrdinal = args.missOrdinal + selectionBase;
    int32_t* missSlot = args.missSlot + selectionBase;
    int32_t* inFlightSlot = args.inFlightSlot + inFlightBase;

    if (threadIdx.x == 0)
    {
        numMiss = 0;
        numCopy = 0;
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

    // Probe once per required unit. A hit resolves immediately; a miss is recorded against
    // its position in the virtual list and resolved once a victim is chosen.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < required; i += kThreads)
    {
        int32_t const ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
        if (ord < 0)
        {
            outSlot[i - inFlight] = kNoSparseSlot;
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
            (i < inFlight ? inFlightSlot[i] : outSlot[i - inFlight]) = slot;
            cls[slot] = kClassHit;
        }
        else
        {
            int32_t const j = atomicAdd(&numMiss, 1);
            missOrd[j] = ord;
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
        bool const pinned = !free && ord >= historyUnits;
        scratch[p] = (free || (cls[slot] == kClassUntouched && !pinned)) ? 1 : 0;
    }
    __syncthreads();

    // Victims come from the tail, least recently used first, so rank from the tail.
    int32_t const totalEvictable = blockExclusiveSum(scratch, entries, temp);
    // The reservation covers a full selection plus every in-flight unit, and the in-flight
    // ones are resolved first, so there is always a cell for each remaining miss.
    KVCM2_SPARSE_REQUIRE(totalEvictable >= numMiss);
    __syncthreads();

    for (int32_t p = static_cast<int32_t>(threadIdx.x); p < entries; p += kThreads)
    {
        int32_t const slot = lruOrder[p];
        int32_t const ord = residentOrdinal[slot];
        bool const free = ord == kFreeOrdinal;
        bool const pinned = !free && ord >= historyUnits;
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
            int32_t const ordinal = missOrd[j];
            int32_t const position = missIdx[j];
            residentOrdinal[slot] = ordinal;
            (position < inFlight ? inFlightSlot[position] : outSlot[position - inFlight]) = slot;
            cls[slot] = kClassFetched;
            // Only history has anything to fetch. An in-flight unit's host page is not
            // written until the watermark passes it, so copying from it would read stale
            // bytes over the data the model is about to write.
            if (ordinal < historyUnits)
            {
                int32_t const c = atomicAdd(&numCopy, 1);
                missOrdinal[c] = ordinal;
                missSlot[c] = slot;
            }
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
        args.missCount[row] = numCopy;
    }
}

//! Copy `bytes` from `src` to `dst`, 16 bytes at a time when every operand allows it.
//! Cold reads cross PCIe, so the wide path is what makes the copy worth doing on the SM.
__device__ inline void copyChunk(char* dst, char const* src, uint32_t bytes)
{
    bool const wide = (reinterpret_cast<uintptr_t>(dst) % 16U == 0U) && (reinterpret_cast<uintptr_t>(src) % 16U == 0U)
        && (bytes % 16U == 0U);
    if (wide)
    {
        auto* dst16 = reinterpret_cast<uint4*>(dst);
        auto const* src16 = reinterpret_cast<uint4 const*>(src);
        uint32_t const count = bytes / 16U;
        for (uint32_t i = threadIdx.x; i < count; i += kThreads)
        {
            dst16[i] = src16[i];
        }
    }
    else
    {
        for (uint32_t i = threadIdx.x; i < bytes; i += kThreads)
        {
            dst[i] = src[i];
        }
    }
}

//! One CTA per (row, buffer, work item).
__global__ __launch_bounds__(kThreads) void sparseUnitCopyKernel(SparseUnitCopyArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.z);
    SparseCacheView const view = args.views[row];
    if (view.residentOrdinal == nullptr)
    {
        // A free row, or a member that has suspended: no reservation to copy into.
        return;
    }

    // The grid walks this call's buffers; the work list is indexed by their layer, which is
    // shared whenever a layer has more than one buffer.
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.y);
    auto const workRow
        = static_cast<ptrdiff_t>(row) * args.workRowStride + (view.bufferLayer[buffer] - args.workRowBase);
    int32_t const j = static_cast<int32_t>(blockIdx.x);
    if (j >= args.workCount[workRow])
    {
        return;
    }

    auto const index = workRow * args.workWidth + j;
    int32_t const ordinal = args.workOrdinal[index];
    int32_t const slot = args.workSlot[index];

    // A unit ordinal names a unit within a block; a cache entry names a unit within a
    // sub-slot. Both divide by the same expansion.
    int32_t const block = ordinal / args.expansion - args.pageTableBase;
    int32_t const srcUnit = ordinal % args.expansion;
    int32_t const dstUnit = slot % args.expansion;
    if (block < 0 || block >= args.numBlocks)
    {
        return;
    }

    // A cell is reached by arithmetic from the lease's base: the arena hands out a
    // contiguous run, so there is no page to look up.
    auto const cell = static_cast<uint64_t>(view.baseCell + buffer * args.cellsPerBuffer + slot / args.expansion);

    auto const hostPage
        = static_cast<uint64_t>(args.pageTable[static_cast<ptrdiff_t>(row) * args.pageTableStride + block]);

    // The cold side is reached by ordinal through the page table, the hot side by entry
    // index through the cache page list. Direction only chooses which one is read.
    auto* cold = reinterpret_cast<char*>(args.hostBase + hostPage * args.hostPageStride + view.pageBufferOffset[buffer]
        + static_cast<uint64_t>(srcUnit) * args.layout.unitStride);
    auto* hot = reinterpret_cast<char*>(
        args.deviceBase + cell * args.bufferBytes + static_cast<uint64_t>(dstUnit) * args.layout.unitStride);
    bool const fetching = args.direction == SparseCopyDirection::kFetch;
    char const* src = fetching ? cold : hot;
    char* dst = fetching ? hot : cold;

    for (uint32_t c = 0; c < args.layout.numChunks; ++c)
    {
        auto const offset = static_cast<uint64_t>(c) * args.layout.chunkStride;
        copyChunk(dst + offset, src + offset, args.layout.chunkBytes);
    }
}

//! One CTA per (row, buffer).
__global__ __launch_bounds__(kThreads) void sparseOutIndexKernel(SparseOutIndexArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.x);
    auto* out = args.out
        + (static_cast<ptrdiff_t>(row) * args.numBuffers + blockIdx.x) * static_cast<ptrdiff_t>(args.selectionWidth);

    SparseCacheView const view = args.views[row];
    bool const reserved = view.residentOrdinal != nullptr;
    auto const base = static_cast<ptrdiff_t>(row) * args.selectionWidth;

    // Every slot is written, so a stale index from a previous step can never be read back.
    // One pass does it, because the output is the selection's own shape: a padded slot and
    // an unreserved row both just take the sentinel here rather than needing a prior fill.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < args.selectionWidth; i += kThreads)
    {
        int32_t const slot = reserved ? args.outSlot[base + i] : kNoSparseSlot;
        if (args.selected[base + i] < 0 || slot < 0)
        {
            out[i] = kBadPageIndexValue;
            continue;
        }

        int32_t const cell = view.baseCell + buffer * args.cellsPerBuffer + slot / args.expansion;
        out[i] = cell * args.expansion + slot % args.expansion;
    }
}

//! One CTA per layer. The units wanted form a contiguous ordinal range, so a scan of the
//! reverse map finds them without a probe index.
__global__ __launch_bounds__(kThreads) void sparseFlushListKernel(SparseFlushListArgs args)
{
    int32_t const layer = static_cast<int32_t>(blockIdx.x);
    int32_t const entries = args.cache.entriesPerLayer;
    int32_t const from = *args.fromUnits;
    int32_t const to = *args.toUnits;

    __shared__ int32_t count;
    if (threadIdx.x == 0)
    {
        count = 0;
    }
    __syncthreads();

    int32_t const* residentOrdinal = args.cache.residentOrdinal + static_cast<ptrdiff_t>(layer) * entries;
    int32_t* outOrdinal = args.outOrdinal + static_cast<ptrdiff_t>(layer) * args.maxFlushUnits;
    int32_t* outSlot = args.outSlot + static_cast<ptrdiff_t>(layer) * args.maxFlushUnits;

    for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
    {
        int32_t const ord = residentOrdinal[s];
        if (ord < from || ord >= to)
        {
            continue;
        }
        int32_t const j = atomicAdd(&count, 1);
        if (j < args.maxFlushUnits)
        {
            outOrdinal[j] = ord;
            outSlot[j] = s;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        // A resident unit the watermark crossed must have been in flight before the move,
        // so the crossed set is bounded by the in-flight headroom this was sized from.
        KVCM2_SPARSE_REQUIRE(count <= args.maxFlushUnits);
        args.outCount[layer] = count;
    }
}

} // namespace

void launchSparseOutIndices(SparseOutIndexArgs const& args, CUstream stream)
{
    if (args.numRows <= 0 || args.numBuffers <= 0 || args.selectionWidth <= 0)
    {
        throw std::invalid_argument("launchSparseOutIndices: every extent must be positive");
    }
    if (args.views == nullptr || args.selected == nullptr || args.outSlot == nullptr || args.out == nullptr)
    {
        throw std::invalid_argument("launchSparseOutIndices: every device array must be provided");
    }
    if (args.cellsPerBuffer <= 0 || args.expansion <= 0 || args.bufferBase < 0)
    {
        throw std::invalid_argument("launchSparseOutIndices: the cell geometry must be positive");
    }

    dim3 const grid(static_cast<unsigned>(args.numBuffers), static_cast<unsigned>(args.numRows));
    sparseOutIndexKernel<<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(args);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream)
{
    if (args.cache.numLayers <= 0 || args.cache.entriesPerLayer <= 0 || args.maxFlushUnits <= 0)
    {
        throw std::invalid_argument("launchSparseFlushList: layers, entries and flush bound must be positive");
    }
    if (args.fromUnits == nullptr || args.toUnits == nullptr || args.outOrdinal == nullptr || args.outSlot == nullptr
        || args.outCount == nullptr)
    {
        throw std::invalid_argument("launchSparseFlushList: every device array must be provided");
    }
    sparseFlushListKernel<<<args.cache.numLayers, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(args);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream)
{
    if (args.numBuffers <= 0 || args.workWidth <= 0 || args.expansion <= 0 || args.numRows <= 0
        || args.workRowStride <= 0)
    {
        throw std::invalid_argument(
            "launchSparseUnitCopy: rows, buffers, work width, work row stride and expansion must be positive");
    }
    if (args.bufferBase < 0 || args.workRowBase < 0 || args.pageTableStride < args.numBlocks)
    {
        throw std::invalid_argument(
            "launchSparseUnitCopy: the bases must be non-negative and the page table stride must cover its blocks");
    }
    if (args.layout.numChunks == 0 || args.layout.chunkBytes == 0)
    {
        throw std::invalid_argument("launchSparseUnitCopy: the unit layout must describe a non-empty unit");
    }
    if (args.pageTable == nullptr || args.views == nullptr || args.workOrdinal == nullptr || args.workSlot == nullptr
        || args.workCount == nullptr)
    {
        throw std::invalid_argument("launchSparseUnitCopy: every device array must be provided");
    }
    if (args.cellsPerBuffer <= 0 || args.bufferBytes == 0)
    {
        throw std::invalid_argument("launchSparseUnitCopy: the cell geometry must be positive");
    }

    dim3 const grid(static_cast<unsigned>(args.workWidth), static_cast<unsigned>(args.numBuffers),
        static_cast<unsigned>(args.numRows));
    sparseUnitCopyKernel<<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(args);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

size_t sparseResolveSharedBytes(int32_t entriesPerLayer, int32_t selectionWidth, int32_t maxInFlightUnits)
{
    return smemLayout(nextPow2(2 * entriesPerLayer), entriesPerLayer, maxInFlightUnits + selectionWidth).total;
}

int32_t sparseResolveMaxEntries(int32_t selectionWidth, int32_t maxInFlightUnits, CUdevice device)
{
    int limit = 0;
    cuCheck(cuDeviceGetAttribute(&limit, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
    int32_t best = 0;
    for (int32_t entries = 1; entries <= (1 << 20); entries <<= 1)
    {
        if (sparseResolveSharedBytes(entries, selectionWidth, maxInFlightUnits) > static_cast<size_t>(limit))
        {
            break;
        }
        best = entries;
    }
    return best;
}

void launchSparseResolve(SparseResolveArgs const& args, CUstream stream)
{
    if (args.numLayers <= 0 || args.entriesPerLayer <= 0 || args.selectionWidth <= 0 || args.numRows <= 0)
    {
        throw std::invalid_argument("launchSparseResolve: rows, layers, entries and selection width must be positive");
    }
    if (args.layerInGroup < 0 || args.layerInGroup >= args.numLayers)
    {
        throw std::invalid_argument("launchSparseResolve: layerInGroup must name a layer of the group");
    }
    if (args.views == nullptr || args.selected == nullptr || args.outSlot == nullptr || args.missOrdinal == nullptr
        || args.missSlot == nullptr || args.missCount == nullptr || args.historyTokens == nullptr
        || args.capacityTokens == nullptr || (args.maxInFlightUnits > 0 && args.inFlightSlot == nullptr))
    {
        throw std::invalid_argument("launchSparseResolve: every device array must be provided");
    }
    if (args.maxInFlightUnits < 0)
    {
        throw std::invalid_argument("launchSparseResolve: maxInFlightUnits must not be negative");
    }
    if (args.unitTokens <= 0)
    {
        throw std::invalid_argument("launchSparseResolve: unitTokens must be positive");
    }

    // entriesPerLayer belongs to the layer group, so one size and one grid serve every row.
    int32_t const tableSize = nextPow2(2 * args.entriesPerLayer);
    size_t const shared = sparseResolveSharedBytes(args.entriesPerLayer, args.selectionWidth, args.maxInFlightUnits);

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
    sparseResolveKernel<<<args.numRows, kThreads, shared, reinterpret_cast<cudaStream_t>(stream)>>>(args, tableSize);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
