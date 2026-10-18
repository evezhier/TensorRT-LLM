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

#pragma once

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storage/config.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SparseCacheView — the device-resident state of one request's sparse history
// cache for one life cycle, as the fetch kernel sees it.
//
// A reserved slot is a row of buffer-sized cells, and any layer's unit may occupy any cell;
// a cell holds `unitsPerPage` units when a selection unit is smaller than a buffer. An entry
// is identified by (layer, entry index) and resolved to a cell arithmetically, so nothing
// ties a layer to a fixed position.
//
// The metadata is laid out [layer][entry]. It is per LAYER, not per buffer: every sparse
// buffer of a layer selects the same units, so one resident set and one recency order serve
// them all, and the entry index the resolve produces addresses each buffer's own cell. That
// makes the agreement between a layer's buffers structural rather than a consequence of
// their running identical logic on identical input.
// ---------------------------------------------------------------------------
struct SparseCacheView
{
    //! Resident unit ordinal per entry, kBadBlockOrdinal when the entry is free.
    //!
    //! Dense rather than keyed: answering "is this ordinal resident, and where" is done by
    //! building a hash over this array in shared memory at the top of each fetch, so the
    //! persistent form stays proportional to cache capacity rather than to context length.
    int32_t* residentOrdinal = nullptr;

    //! Entry indices in recency order, most-recently-used first.
    int32_t* lruOrder = nullptr;

    //! This request's lease on the arena. A cell address is arithmetic from here -- see
    //! "Why the cache is a flat array of cells" in SPARSE_ATTENTION_DESIGN.md -- so there is
    //! no page table to consult. The lease is an index rather than a cell offset because a
    //! pool's region has its own cells-per-lease, which the copy supplies per launch.
    int32_t leaseIndex = 0;

    //! [numBuffers] each buffer's index among those sharing its pool: where its cells sit
    //! inside that pool's arena region.
    int32_t* bufferSlotInPool = nullptr;

    //! [numBuffers] the layer each buffer belongs to, i.e. which metadata row carries its
    //! residency and its work list. The copy walks buffers but reads a list produced per
    //! layer, so it needs the map to join the two. Non-decreasing, because the buffers of a
    //! layer are consecutive, which is what lets a one-layer call name its buffers as a
    //! range rather than a set.
    int32_t* bufferLayer = nullptr;

    //! Layers in this life cycle: the metadata's first dimension.
    int32_t numLayers = 0;
    //! Sparse buffers in this life cycle, one per (layer, role). Cells and page offsets are
    //! per buffer, so this is the dimension the copy walks.
    int32_t numBuffers = 0;
    int32_t entriesPerLayer = 0;

    [[nodiscard]] constexpr int32_t numEntries() const noexcept
    {
        return numLayers * entriesPerLayer;
    }
};

// ---------------------------------------------------------------------------
// SparseCacheInit — everything a request's cache state is built from, all of it
// fixed for the life of the reservation.
// ---------------------------------------------------------------------------
struct SparseCacheInit
{
    int32_t numLayers = 0;
    int32_t numBuffers = 0;
    int32_t entriesPerLayer = 0;
    //! Cells one buffer gets, i.e. entriesPerLayer / unitsPerPage. A lease is this many cells
    //! per buffer, laid out buffer-major.
    int32_t cellsPerBuffer = 0;
    //! [numBuffers] the layer each buffer belongs to, non-decreasing.
    std::vector<int32_t> bufferLayer;
    //! [numBuffers] each buffer's index among those sharing its pool.
    std::vector<int32_t> bufferSlotInPool;
    //! One entry per pool of the life cycle: cell size, and buffers drawing on it.
    std::vector<std::pair<uint64_t, int32_t>> pools;
};

// ---------------------------------------------------------------------------
// SparseCacheLease — one request's cells and the metadata describing them.
//
// Taken at resume() and returned at suspend() or close(). Move-only and RAII: the arena's
// free list is restored by the destructor, so a request dropped on any path -- including a
// poisoned teardown -- cannot leak its cells.
// ---------------------------------------------------------------------------
class SparseCellArena;

class SparseCacheLease
{
public:
    SparseCacheLease() = default;
    ~SparseCacheLease();

    SparseCacheLease(SparseCacheLease&& other) noexcept;
    SparseCacheLease& operator=(SparseCacheLease&& other) noexcept;
    SparseCacheLease(SparseCacheLease const&) = delete;
    SparseCacheLease& operator=(SparseCacheLease const&) = delete;

    [[nodiscard]] bool empty() const noexcept
    {
        return mArena == nullptr;
    }

    [[nodiscard]] SparseCacheView const& view() const noexcept
    {
        return mView;
    }

    //! A device copy of view(), as a one-row array.
    //!
    //! The copy kernel reads its per-row state through a view array, so that a batched fetch
    //! can serve rows whose metadata belongs to different requests. A single-request
    //! write-back takes the same path with one row rather than a second addressing mode.
    [[nodiscard]] SparseCacheView const* deviceView() const noexcept
    {
        return mDeviceView;
    }

    //! Return the cells early. Idempotent; the destructor does the same.
    void release() noexcept;

    //! Copy the resident ordinals back to the host. Test and debug use only; synchronises.
    [[nodiscard]] std::vector<int32_t> readResidentOrdinals() const;

    //! Copy the per-layer write-back counts back to the host, -1 where no write-back has
    //! run. Test and debug use only; synchronises.

private:
    friend class SparseCellArena;

    SparseCellArena* mArena = nullptr;
    int mIndex = -1;
    SparseCacheView mView;
    SparseCacheView* mDeviceView = nullptr;
};

// ---------------------------------------------------------------------------
// SparseCellArena — one life cycle's GPU cells, and the metadata that describes them.
//
// A dedicated allocation rather than storage slots. That is what lets the life cycle's hot
// pool group be host memory without the cache following it there, and it frees the cache
// from the allocator's coalesced-slot granularity: every request of a life cycle needs
// exactly the same number of cells, so this is a fixed-size-block free list.
//
// One block holds the cells of every lease followed by their metadata. Cells come first so
// that a cell index is relative to the arena's base, which is what the fetch reports and
// what get_mem_pool_base_address returns for a sparse buffer.
// ---------------------------------------------------------------------------
class SparseCellArena
{
public:
    SparseCellArena() = default;

    //! @param shape     per-request metadata and cell geometry, fixed by configuration,
    //!                   including one entry per pool of the life cycle.
    //! @param maxLeases  concurrent requests this life cycle can serve.
    //!
    //! One region per pool, mirroring the hot pool group: pools partition the life cycle's
    //! buffers by size, and a cell is one buffer, so cells cannot be uniform across them.
    SparseCellArena(SparseCacheInit const& shape, int maxLeases);

    ~SparseCellArena() = default;
    SparseCellArena(SparseCellArena&&) noexcept = default;
    SparseCellArena& operator=(SparseCellArena&&) noexcept = default;
    SparseCellArena(SparseCellArena const&) = delete;
    SparseCellArena& operator=(SparseCellArena const&) = delete;

    [[nodiscard]] bool empty() const noexcept
    {
        return mMemory == nullptr;
    }

    //! Base of one pool's cell region. Indices the fetch emits for a buffer are relative to
    //! its pool's base, which is what get_sparse_cache_base_address reports for that buffer.
    [[nodiscard]] PoolIndex numPools() const noexcept
    {
        return mPoolBase.size();
    }

    [[nodiscard]] CUdeviceptr base(PoolIndex pool) const
    {
        return mPoolBase.at(pool);
    }

    [[nodiscard]] uint64_t bufferBytes(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).first;
    }

    //! Cells one lease takes in this pool's region: its buffers times the cells each gets.
    [[nodiscard]] int cellsPerLease(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).second * mShape.cellsPerBuffer;
    }

    [[nodiscard]] int freeLeases() const noexcept
    {
        return static_cast<int>(mFree.size());
    }

    //! Device bytes this arena holds: every pool's cell region, the state its leases share,
    //! and one metadata block per lease. The whole allocation, so a caller reserving the
    //! arena's bytes against a budget can check its own figure against this one.
    [[nodiscard]] size_t numBytes() const noexcept
    {
        return mNumBytes;
    }

    //! Take a lease, enqueuing its metadata reset on `stream`. nullopt when the arena is
    //! full, which a resume reports as a failed reservation rather than throwing.
    //!
    //! Entries start free: every ordinal is kBadBlockOrdinal and the recency order is the
    //! identity permutation.
    [[nodiscard]] std::optional<SparseCacheLease> acquire(CUstream stream);

    //! [numBuffers] the layer each buffer belongs to, as every view of this life cycle sees
    //! it. Reachable without a view because the copy has to pick a layer's work list before
    //! it knows which row an entry names.
    [[nodiscard]] int32_t const* bufferLayer() const noexcept
    {
        return mSharedLayer;
    }

    //! The shape this arena was built from, for a caller folding its per-buffer constants
    //! into a descriptor rather than reading them through a view at every work item.
    [[nodiscard]] SparseCacheInit const& shape() const noexcept
    {
        return mShape;
    }

private:
    friend class SparseCacheLease;

    void _release(int index) noexcept;

    CudaUniqPtr<> mMemory;
    SparseCacheInit mShape;
    size_t mMetadataBytes = 0;
    size_t mNumBytes = 0;
    TypedVec<PoolIndex, CUdeviceptr> mPoolBase;
    std::byte* mMetadataBase = nullptr;

    //! Shared by every lease of the life cycle, so a resume copies neither.
    int32_t* mSharedLayer = nullptr;
    int32_t* mSharedSlotInPool = nullptr;
    int32_t* mIdentity = nullptr;

    std::vector<int> mFree;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
