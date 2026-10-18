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
#include "tensorrt_llm/common/envUtils.h"

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
constexpr uint32_t kWarpSize = 32;
//! Warps in one copy CTA. The CTA size is chosen for occupancy; how many warps serve one
//! row is a separate number, so the two are decoupled by remapping a global warp id.
constexpr int32_t kWarpsPerCopyCta = kThreads / static_cast<int>(kWarpSize);
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

//! Programmatic dependent launch, where the hardware has it.
//!
//! Both calls are hints, not contracts: a wait with no launch dependency returns at once,
//! and a release with no dependent kernel does nothing. So a kernel carrying them is correct
//! whether or not its launch enrolled in PDL, and compiling them out below Hopper only falls
//! back to ordinary stream ordering.
__device__ inline void gridDependencyWait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

//! Let a dependent grid start. Only sound once everything it reads from this one is written.
__device__ inline void gridDependencyRelease()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
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
    __shared__ int32_t numResolved;

    // The view's own arrays belong to this row's allocation, so only the layer indexes them.
    int32_t* residentOrdinal = view.residentOrdinal + static_cast<ptrdiff_t>(layer) * entries;
    int32_t* lruOrder = view.lruOrder + static_cast<ptrdiff_t>(layer) * entries;
    // The batch-wide arrays are indexed by row alone: the call is one layer, and its
    // buffers share the single result.
    auto const selectionBase = static_cast<ptrdiff_t>(row) * width;
    auto const inFlightBase = static_cast<ptrdiff_t>(row) * args.maxInFlightUnits;
    int32_t const* selected = args.selected + selectionBase;
    int32_t* outSlot = args.outSlot + selectionBase;
    int32_t* inFlightSlot = args.inFlightSlot + inFlightBase;

    if (threadIdx.x == 0)
    {
        numMiss = 0;
        numCopy = 0;
        numResolved = 0;
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
    //
    // Above the dependency wait deliberately: this reads the layer's own residency table and
    // writes only shared memory, and nothing that runs before a resolve touches that table --
    // a previous layer's resolve owns a different slice of it, and the copy and index kernels
    // between them write cells and indices. So the whole index build is work a block can
    // finish while the kernel it waits on is still draining.
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

    // The selection is the caller's, produced by whatever chose the units this step wants,
    // so this is as far as a block gets without it. Every global write stays below the wait
    // too: a block that has not yet waited may still be running beside a grid that could
    // read what it would overwrite.
    gridDependencyWait();

    // Slots past the in-flight span say "no unit in flight", so a consumer can read this
    // array without also being told how many units there are.
    for (int32_t i = static_cast<int32_t>(threadIdx.x) + inFlight; i < args.maxInFlightUnits; i += kThreads)
    {
        inFlightSlot[i] = kNoSparseSlot;
    }
    __syncthreads();

    // Probe once per required unit. A hit resolves immediately; a miss is recorded against
    // its position in the virtual list and resolved once a victim is chosen.
    //
    // Selection entries naming real history are counted as they pass: they are the
    // denominator of the fetch's hit rate, and only this pass knows which they are. A
    // padded entry is neither hit nor miss, and an in-flight unit is not a fetch -- its
    // host page is not written until the watermark passes it.
    int32_t resolvedMine = 0;
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < required; i += kThreads)
    {
        int32_t const ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
        if (ord < 0)
        {
            outSlot[i - inFlight] = kNoSparseSlot;
            continue;
        }
        resolvedMine += i < inFlight ? 0 : 1;
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
    if (resolvedMine > 0)
    {
        atomicAdd(&numResolved, resolvedMine);
    }
    __syncthreads();

    // Claim this row's run in the compacted list now that the miss set is final, so the
    // eviction loop below can write straight into it. Only a miss below the watermark is
    // copied, which is the same test the loop applies, so counting it here costs one pass
    // over a list already in shared memory and saves staging the entries anywhere.
    __shared__ int32_t copyBase;
    {
        int32_t mine = 0;
        for (int32_t j = static_cast<int32_t>(threadIdx.x); j < numMiss; j += kThreads)
        {
            mine += missOrd[j] < historyUnits ? 1 : 0;
        }
        int32_t scanned = 0;
        int32_t claimed = 0;
        BlockScan(temp).ExclusiveSum(mine, scanned, claimed);
        if (threadIdx.x == 0)
        {
            copyBase = claimed > 0 ? atomicAdd(args.work.total, claimed) : 0;
            // The list is sized for every row missing its whole selection, so a claim can
            // only run past it if the caller sized it from a different width.
            KVCM2_SPARSE_REQUIRE(copyBase + claimed <= args.work.capacity);
            if (args.hitMiss != nullptr)
            {
                // This row's own contribution, not a total read back later: the step's
                // counters are cleared per publish, not per fetch, so anything reading them
                // afterwards would count a second fetch of the same layer twice.
                atomicAdd(&args.hitMiss->resolved, static_cast<unsigned long long>(numResolved));
                atomicAdd(&args.hitMiss->misses, static_cast<unsigned long long>(claimed));
            }
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
                auto const c = copyBase + atomicAdd(&numCopy, 1);
                args.work.row[c] = row;
                args.work.ordinal[c] = ordinal;
                args.work.slot[c] = slot;
            }
        }
    }
    __syncthreads();

    // The work list and the slots are final here; the recency rebuild below is this
    // kernel's own bookkeeping, which nothing downstream reads. Releasing now hands the
    // copy the whole rebuild as overlap.
    gridDependencyRelease();

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

//! Copy `bytes` from `src` to `dst` with one warp, `sizeof(Vec)` at a time.
//!
//! The width is the launcher's choice, made from the layout and the addresses it produces,
//! so there is no alignment test here and no tail: `bytes` divides by `sizeof(Vec)` and both
//! pointers are aligned to it. Cold reads cross PCIe, which is what makes the wide types
//! worth choosing when a layout admits them.
template <typename Vec>
__device__ inline void copyChunkWarp(char* dst, char const* src, uint32_t bytes, uint32_t lane)
{
    auto* out = reinterpret_cast<Vec*>(dst);
    auto const* in = reinterpret_cast<Vec const*>(src);
    uint32_t const count = bytes / static_cast<uint32_t>(sizeof(Vec));
    for (uint32_t i = lane; i < count; i += kWarpSize)
    {
        out[i] = in[i];
    }
}

//! One pool's work, with a vector width fixed at compile time.
//!
//! Split from the kernel so the width can be switched on once per block rather than
//! templated into four kernels: with the pools merged into one launch they no longer each
//! have a launch to be templated by.
template <typename Vec>
__device__ void copySparseUnits(SparseUnitCopyArgs const& args, SparseCopyDesc const& desc, int32_t list, int32_t first)
{
    uint32_t const lane = threadIdx.x % kWarpSize;
    int32_t const stride = static_cast<int32_t>(gridDim.x) * kWarpsPerCopyCta;
    auto const listBase = static_cast<ptrdiff_t>(list) * args.work.capacity;
    int32_t const count = args.work.total[list];

    for (int32_t j = first; j < count; j += stride)
    {
        auto const index = listBase + j;
        int32_t const row = args.work.row[index];
        int32_t const ordinal = args.work.ordinal[index];
        int32_t const slot = args.work.slot[index];

        // Read per item rather than per warp: the list is dense across rows, so consecutive
        // entries need not share one. A producer's CTA claims a contiguous run, so in
        // practice a warp's items mostly do.
        SparseCacheView const view = args.views[row];

        // A unit ordinal names a unit within a block; a cache entry names a unit within a
        // sub-slot. Both divide by the same units-per-page.
        int32_t const block = ordinal / args.unitsPerPage;
        int32_t const srcUnit = ordinal % args.unitsPerPage;
        int32_t const dstUnit = slot % args.unitsPerPage;
        if (block >= static_cast<int32_t>(desc.numBlocks))
        {
            // Outside the extent this call was given; the warp's other items may not be.
            continue;
        }

        // The descriptor's pointer already stands at this buffer's entry, so the table
        // index is the row's run of blocks and nothing else.
        auto const hostPage
            = static_cast<uint64_t>(desc.pageTable[static_cast<ptrdiff_t>(row) * desc.pageTableStride + block]);

        // The cold side is reached by ordinal through the page table, the hot side by
        // arithmetic from the lease: the arena hands out a contiguous run, so there is no
        // page to look up, and the buffer's own slot in the pool is already in deviceBase.
        // Direction only chooses which of the two is read.
        auto* cold = reinterpret_cast<char*>(
            desc.hostBase + hostPage * desc.hostPageStride + static_cast<uint64_t>(srcUnit) * desc.layout.unitStride);
        auto* hot = reinterpret_cast<char*>(desc.deviceBase + static_cast<uint64_t>(view.leaseIndex) * desc.leaseStride
            + static_cast<uint64_t>(slot / args.unitsPerPage) * desc.cellBytes
            + static_cast<uint64_t>(dstUnit) * desc.layout.unitStride);
        bool const fetching = args.direction == SparseCopyDirection::kFetch;
        char const* src = fetching ? cold : hot;
        char* dst = fetching ? hot : cold;

        for (uint32_t c = 0; c < desc.layout.numChunks; ++c)
        {
            auto const offset = static_cast<uint64_t>(c) * desc.layout.chunkStride;
            copyChunkWarp<Vec>(dst + offset, src + offset, desc.layout.chunkBytes, lane);
        }
    }
}

//! One warp per work item, walking the compacted list with a grid-stride loop. `blockIdx.z`
//! is the pool, `blockIdx.y` its buffer, `blockIdx.x` the work.
__global__ __launch_bounds__(kThreads) void sparseUnitCopyKernel(SparseUnitCopyArgs args)
{
    // Broadcast from lane zero before anything can diverge: threadIdx.x varies within the
    // warp, so the compiler cannot otherwise see that this quotient does not, and every lane
    // must still be present for the shuffle.
    uint32_t warpInCta = threadIdx.x / kWarpSize;
    warpInCta = __shfl_sync(0xffffffffU, warpInCta, 0);
    int32_t const globalWarp = static_cast<int32_t>(blockIdx.x) * kWarpsPerCopyCta + static_cast<int32_t>(warpInCta);

    // Everything down to the dependency wait is the producer's business only in that it
    // names where to put the result: the descriptors are built once with the Batch, and the
    // buffer and list follow from the block's own coordinates. Resolving it here is what the
    // launch dependency is for -- the block is resident and addressed before the kernel it
    // waits on has finished.
    //
    // One flat buffer range rather than pools by buffers-per-pool, so a pool narrower than
    // its neighbours costs no idle blocks, and one descriptor load rather than a map into a
    // per-pool table: everything that would have been looked up is folded in at build time.
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.y);
    SparseCopyDesc const desc = args.copyDesc[buffer];
    // The buffer is this block's, so its layer -- and therefore its list -- is uniform and
    // known before any entry is read. The list index is zero whenever the caller passed one
    // list, which is what a fetch does: it resolves a single layer.
    int32_t const list = args.work.numLists > 1 ? args.bufferLayer[buffer] - args.work.layerBase : 0;

    // The work list and its total are the producer's, so this is as far as the block gets
    // before the kernel that fills them has run.
    gridDependencyWait();

    // Nothing launched after this reads a cell, so a successor need not wait for the copy
    // itself -- only for the block to have got this far. Releasing here is what keeps the
    // index kernel that follows on the stream from being ordered behind a copy it does not
    // depend on.
    gridDependencyRelease();

    switch (desc.width)
    {
    case SparseCopyWidth::kWord4: copySparseUnits<uint4>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kWord2: copySparseUnits<uint2>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kWord: copySparseUnits<uint32_t>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kByte: copySparseUnits<char>(args, desc, list, globalWarp); break;
    }
}

//! One CTA per (row, buffer).
__global__ __launch_bounds__(kThreads) void sparseOutIndexKernel(SparseOutIndexArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.x);
    auto* out = args.out
        + (static_cast<ptrdiff_t>(row) * args.outBuffers + args.outBase + blockIdx.x)
            * static_cast<ptrdiff_t>(args.selectionWidth);

    SparseCacheView const view = args.views[row];
    bool const reserved = view.residentOrdinal != nullptr;
    auto const base = static_cast<ptrdiff_t>(row) * args.selectionWidth;
    int32_t const slotStride = args.outSlotStride > 0 ? args.outSlotStride : args.selectionWidth;
    auto const slotBase = static_cast<ptrdiff_t>(row) * slotStride;

    // Every slot is written, so a stale index from a previous step can never be read back.
    // One pass does it, because the output is the selection's own shape: a padded slot and
    // an unreserved row both just take the sentinel here rather than needing a prior fill.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < args.selectionWidth; i += kThreads)
    {
        int32_t const slot = reserved ? args.outSlot[slotBase + i] : kNoSparseSlot;
        // Without a selection array the slot alone says whether the entry is live, which is
        // what the write-index path passes: its in-flight list is already self-describing.
        if ((args.selected != nullptr && args.selected[base + i] < 0) || slot < 0)
        {
            out[i] = kBadPageIndexValue;
            continue;
        }

        int32_t const cell = view.leaseIndex * args.cellsPerLease + view.bufferSlotInPool[buffer] * args.cellsPerBuffer
            + slot / args.unitsPerPage;
        out[i] = cell * args.unitsPerPage + slot % args.unitsPerPage;
    }
}

//! One CTA per layer. The units wanted form a contiguous ordinal range, so a scan of the
//! reverse map finds them without a probe index.
__global__ __launch_bounds__(kThreads) void sparseFlushListKernel(SparseFlushListArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    SparseCacheView const view = args.views[row];
    if (view.residentOrdinal == nullptr)
    {
        // A free row, or a member that has suspended: no reservation to write back from.
        return;
    }
    int32_t const layer = static_cast<int32_t>(blockIdx.x);
    int32_t const entries = view.entriesPerLayer;

    // The row's view is staged with the watermarks and is nobody's output here, so a block
    // can address itself before waiting. The range itself is uploaded by the call that asked
    // for this write-back, so reading it is where the block has to stop.
    gridDependencyWait();

    int32_t const from = args.fromUnits[row];
    int32_t const to = args.toUnits[row];
    if (from >= to)
    {
        // This row's watermark did not move, so nothing of its crossed, but the grid is
        // sized for the widest row.
        if (threadIdx.x == 0)
        {
            args.outCount[static_cast<ptrdiff_t>(row) * args.numLayers + layer] = 0;
        }
        return;
    }

    __shared__ typename BlockScan::TempStorage temp;
    int32_t const* residentOrdinal = view.residentOrdinal + static_cast<ptrdiff_t>(layer) * entries;
    auto const crossed = [&](int32_t slot)
    {
        int32_t const ord = residentOrdinal[slot];
        return ord >= from && ord < to;
    };

    // Two passes over the resident set, because the run cannot be claimed until its length
    // is known and the entries are too many to stage in shared memory: a prefilling row
    // crosses as many units as its prompt holds. The set is the per-request cache, which is
    // small enough that scanning it twice costs less than staging it once.
    __shared__ int32_t count;
    __shared__ int32_t base;
    if (threadIdx.x == 0)
    {
        count = 0;
    }
    __syncthreads();

    {
        int32_t mine = 0;
        for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
        {
            mine += crossed(s) ? 1 : 0;
        }
        int32_t scanned = 0;
        int32_t claimed = 0;
        BlockScan(temp).ExclusiveSum(mine, scanned, claimed);
        if (threadIdx.x == 0)
        {
            base = claimed > 0 ? atomicAdd(args.work.total + layer, claimed) : 0;
            // Sized from the step's token budget, which bounds how far every row's watermark
            // can move together. Running past it means the budget the Batch was given is not
            // the one the caller honoured.
            KVCM2_SPARSE_REQUIRE(base + claimed <= args.work.capacity);
            args.outCount[static_cast<ptrdiff_t>(row) * args.numLayers + layer] = claimed;
        }
    }
    __syncthreads();

    auto const listBase = static_cast<ptrdiff_t>(layer) * args.work.capacity + base;
    for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
    {
        if (!crossed(s))
        {
            continue;
        }
        auto const j = listBase + atomicAdd(&count, 1);
        args.work.row[j] = row;
        args.work.ordinal[j] = residentOrdinal[s];
        args.work.slot[j] = s;
    }

    // Unlike the resolve, the list is the last thing this kernel writes, so there is no tail
    // to overlap -- only the copy's launch to bring forward.
    __syncthreads();
    gridDependencyRelease();
}

} // namespace

//! Throw unless the work list is fully described. Shared because all three launchers take
//! the same descriptor, and a half-set one is the easiest mistake to make when adding a
//! caller.
void checkWorkList(SparseWorkList const& work, char const* api)
{
    if (work.row == nullptr || work.ordinal == nullptr || work.slot == nullptr || work.total == nullptr)
    {
        throw std::invalid_argument(std::string(api) + ": the work list's device arrays must be provided");
    }
    if (work.capacity <= 0 || work.numLists <= 0 || work.layerBase < 0)
    {
        throw std::invalid_argument(std::string(api) + ": the work list's capacity and list count must be positive");
    }
}

void launchSparseOutIndices(SparseOutIndexArgs const& args, CUstream stream)
{
    if (args.numRows <= 0 || args.numBuffers <= 0 || args.selectionWidth <= 0)
    {
        throw std::invalid_argument("launchSparseOutIndices: every extent must be positive");
    }
    if (args.views == nullptr || args.outSlot == nullptr || args.out == nullptr)
    {
        throw std::invalid_argument("launchSparseOutIndices: every device array must be provided");
    }
    if (args.cellsPerBuffer <= 0 || args.cellsPerLease <= 0 || args.unitsPerPage <= 0 || args.bufferBase < 0)
    {
        throw std::invalid_argument("launchSparseOutIndices: the cell geometry must be positive");
    }
    if (args.outBase < 0 || args.outBase + args.numBuffers > args.outBuffers)
    {
        throw std::invalid_argument("launchSparseOutIndices: this call's buffers must fit the output row");
    }
    if (args.outSlotStride < 0 || (args.outSlotStride > 0 && args.outSlotStride < args.selectionWidth))
    {
        throw std::invalid_argument("launchSparseOutIndices: the slot stride must cover what this call reads");
    }

    dim3 const grid(static_cast<unsigned>(args.numBuffers), static_cast<unsigned>(args.numRows));
    sparseOutIndexKernel<<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(args);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream)
{
    if (args.numRows <= 0 || args.numLayers <= 0)
    {
        throw std::invalid_argument("launchSparseFlushList: rows and layers must be positive");
    }
    if (args.views == nullptr || args.fromUnits == nullptr || args.toUnits == nullptr || args.outCount == nullptr)
    {
        throw std::invalid_argument("launchSparseFlushList: every device array must be provided");
    }
    checkWorkList(args.work, "launchSparseFlushList");
    if (args.work.numLists != args.numLayers)
    {
        throw std::invalid_argument("launchSparseFlushList: a write-back needs one work list per layer");
    }
    dim3 const grid(static_cast<unsigned>(args.numLayers), static_cast<unsigned>(args.numRows));
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "sparseFlushList", sparseFlushListKernel, grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream), args);
}

SparseCopyWidth sparseCopyWidth(SparseUnitLayout const& layout, uint64_t hostPageStride, uint64_t bufferBytes)
{
    // Sizes only. Every offset a lane adds to a base is a sum of these, so a width dividing
    // all of them leaves every access aligned relative to the base -- and the bases
    // themselves are checked once, on the host, where a misaligned pool is a configuration
    // error rather than a case to carry into the kernel.
    auto const fits = [&](uint64_t width)
    {
        auto const divides = [width](uint64_t v) { return v % width == 0; };
        return divides(hostPageStride) && divides(bufferBytes) && divides(layout.unitStride)
            && divides(layout.chunkStride) && divides(layout.chunkBytes);
    };
    if (fits(sizeof(uint4)))
    {
        return SparseCopyWidth::kWord4;
    }
    if (fits(sizeof(uint2)))
    {
        return SparseCopyWidth::kWord2;
    }
    if (fits(sizeof(uint32_t)))
    {
        return SparseCopyWidth::kWord;
    }
    return SparseCopyWidth::kByte;
}

size_t sparseCopyWidthBytes(SparseCopyWidth width) noexcept
{
    switch (width)
    {
    case SparseCopyWidth::kWord4: return sizeof(uint4);
    case SparseCopyWidth::kWord2: return sizeof(uint2);
    case SparseCopyWidth::kWord: return sizeof(uint32_t);
    case SparseCopyWidth::kByte: break;
    }
    return 1;
}

void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream)
{
    if (args.unitsPerPage <= 0 || args.numRows <= 0 || args.gridCtas <= 0 || args.numBuffers <= 0
        || args.bufferBase < 0)
    {
        throw std::invalid_argument(
            "launchSparseUnitCopy: rows, buffers, grid CTAs and units-per-page must be positive");
    }
    if (args.copyDesc == nullptr || args.views == nullptr)
    {
        throw std::invalid_argument("launchSparseUnitCopy: every device array must be provided");
    }
    checkWorkList(args.work, "launchSparseUnitCopy");
    if (args.work.numLists > 1 && args.bufferLayer == nullptr)
    {
        throw std::invalid_argument("launchSparseUnitCopy: a per-layer work list needs the buffer-to-layer map");
    }

    dim3 const grid(static_cast<unsigned>(args.gridCtas), static_cast<unsigned>(args.numBuffers));
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "sparseUnitCopy", sparseUnitCopyKernel, grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream), args);
}

int32_t sparseCopyMultiprocessors(CUdevice device)
{
    int smCount = 0;
    cuCheck(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    return std::max(smCount, 1);
}

int32_t sparseCopyGridCtas(int32_t workCapacity, int32_t smCount) noexcept
{
    //! Resident CTAs per SM this heuristic aims for.
    constexpr int32_t kCtasPerSm = 4;

    // Capped by the work itself: a CTA past the list's capacity could never find an item.
    int32_t const full = std::max(smCount, 1) * kCtasPerSm;
    int32_t const needed = (workCapacity + kWarpsPerCopyCta - 1) / kWarpsPerCopyCta;
    return std::max(1, std::min(full, needed));
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
    if (args.numLayers <= 0 || args.entriesPerLayer <= 0 || args.selectionWidth < 0 || args.numRows <= 0)
    {
        throw std::invalid_argument("launchSparseResolve: rows, layers and entries must be positive");
    }
    // A zero-width resolve is the write-index path: it makes the in-flight unit resident and
    // nothing else, so it needs neither a selection to read nor a miss list to fill.
    if (args.selectionWidth == 0 && args.maxInFlightUnits == 0)
    {
        throw std::invalid_argument("launchSparseResolve: a resolve with no selection must have an in-flight span");
    }
    if (args.layerInGroup < 0 || args.layerInGroup >= args.numLayers)
    {
        throw std::invalid_argument("launchSparseResolve: layerInGroup must name a layer of the group");
    }
    bool const selectionArrays = args.selectionWidth == 0 || (args.selected != nullptr && args.outSlot != nullptr);
    if (args.views == nullptr || !selectionArrays || args.missCount == nullptr || args.historyTokens == nullptr
        || args.capacityTokens == nullptr || (args.maxInFlightUnits > 0 && args.inFlightSlot == nullptr))
    {
        throw std::invalid_argument("launchSparseResolve: every device array must be provided");
    }
    if (args.maxInFlightUnits < 0)
    {
        throw std::invalid_argument("launchSparseResolve: maxInFlightUnits must not be negative");
    }
    if (args.selectionWidth > 0)
    {
        checkWorkList(args.work, "launchSparseResolve");
        if (args.work.numLists != 1)
        {
            throw std::invalid_argument("launchSparseResolve: a fetch resolves one layer, so it fills one work list");
        }
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
    tensorrt_llm::common::launchWithPdlWhenEnabled("sparseResolve", sparseResolveKernel,
        dim3(static_cast<unsigned>(args.numRows)), dim3(kThreads), shared, reinterpret_cast<cudaStream_t>(stream), args,
        tableSize);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
