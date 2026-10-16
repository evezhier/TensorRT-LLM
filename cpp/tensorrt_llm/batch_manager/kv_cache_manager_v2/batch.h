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
#include "kv_cache_manager_v2/lifeCycleRegistry.h"
#include "kv_cache_manager_v2/sparseCache.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

#include <cuda.h>

#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! One layer group's page table as the device sees it. Always packed: the Batch allocates
//! it, so a row is exactly `blocksPerRow` entries.
struct PageTableView
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! Host page-index storage owned by someone else, which a Batch's members write into
//! instead of into staging of its own.
//!
//! The runtime already holds one of these -- the host offsets tensor the dense attention
//! path fills -- and its rows carry exactly the numbers a member would otherwise write
//! somewhere else as well. Since a KvCache can write to only one buffer at a time, having
//! two means handing the binding back and forth every step. Pointing the member at the
//! runtime's row instead leaves one buffer with one writer, and the binding then survives
//! the whole decode.
//!
//! Only the host side is borrowed. The device table stays the Batch's own, because
//! publish() must produce a snapshot: without one the host could not write a row until
//! every kernel that reads it had finished, which for a captured graph means the end of the
//! forward. The lender's own device mirror is no substitute -- it is read by no captured
//! kernel today, which is what lets its upload overlap the previous step, and sharing it
//! would give that up.
//!
//! Rows are laid out `[layer_group][row][block]` with the strides below. A row is usually
//! longer than the Batch's `blocksPerRow`, since the lender's tensor carries a key/value
//! dimension this table does not use.
struct ExternalPageTable
{
    int32_t* host = nullptr;
    //! Elements between consecutive layer groups, and between consecutive rows of one.
    int64_t layerGroupStride = 0;
    int64_t rowStride = 0;
};

//! Device scratch one layer group's fetch resolves into, before the copy and the index
//! scatter read it back.
//!
//! Owned by Batch rather than allocated per call because every extent is a configuration
//! constant -- rows, selection width, in-flight headroom -- so one allocation serves every
//! step, and a step that allocates nothing is a step a CUDA graph can capture.
struct SparseFetchScratch
{
    int32_t* outSlot = nullptr;      //!< [numRows][selectionWidth]
    int32_t* missOrdinal = nullptr;  //!< [numRows][selectionWidth]
    int32_t* missSlot = nullptr;     //!< [numRows][selectionWidth]
    int32_t* missCount = nullptr;    //!< [numRows]
    int32_t* inFlightSlot = nullptr; //!< [numRows][maxInFlightUnits]
    int32_t selectionWidth = 0;
    int32_t maxInFlightUnits = 0;
};

//! Per-row watermarks, in tokens. Everything a step needs about how far a request has got
//! derives from these two, at whatever granularity the consumer works in:
//!
//!     num_blocks     = historyLength / tokens_per_block     (valid page table entries)
//!     history_units  = historyLength / unit_tokens          (selectable units)
//!     capacity_units = capacity / unit_tokens               (the in-flight span's end)
//!
//! Publishing the source rather than the three derived counts keeps one truth; the divisors
//! are per-layer-group constants a kernel already has.
//!
//! A free row reads zero, so it describes a request with nothing eligible.
struct WatermarkView
{
    CUdeviceptr historyLength = 0;
    CUdeviceptr capacity = 0;
    int32_t numRows = 0;
};

// ---------------------------------------------------------------------------
// Batch — the KvCaches stepped together, and their device-side mirror.
//
// Owns the row order and the per-layer-group page tables the sparse fetch reads. Rows are
// slots, not positions in a compacted list: removing a member leaves a hole rather than
// shifting later rows, so array shapes and addresses survive a CUDA graph capture.
//
// Membership is exclusive. A KvCache detaches itself on destruction, so a dropped request
// cannot leave a dangling row.
// ---------------------------------------------------------------------------
class Batch
{
public:
    //! @param maxBatchSize    rows, i.e. the most requests that can be stepped together.
    //! @param maxBlocksPerSeq columns, i.e. the longest sequence a row can describe.
    //!
    //! Both bound device arrays whose shape must not change between steps, so neither can
    //! be grown later. Pass 0 to derive from the manager's `constraints`, which fails when
    //! none were declared -- a pool-sizing hint is not a batch bound, so it is only a
    //! default, never an inference.
    //!
    //! @param pageTable borrowed host page-index storage. When absent the Batch stages its
    //! own, which is what a caller with no runtime tensor to share wants. When present its
    //! rows must be addressable for every (layer group, row) this Batch can reach, and the
    //! rows become the lender's -- so addKvCache needs to be told which one a member
    //! occupies rather than choosing. The device table is the Batch's either way.
    explicit Batch(KvCacheManager& manager, int maxBatchSize = 0, int maxBlocksPerSeq = 0,
        std::optional<ExternalPageTable> pageTable = std::nullopt);
    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Seat a cache in `row`, or in the lowest free row when it is negative.
    //!
    //! A borrowed page table makes the row the owner's to choose: its rows are indexed by
    //! whatever slot the owner assigned the request, and a row this Batch picked for itself
    //! would address a different request's entries. So `row` is required in that case and
    //! refused in neither -- passing one is always allowed.
    //!
    //! Throws when the row is taken or out of range, the batch is full, the cache already
    //! belongs to a batch, or its beam width exceeds one -- only beam 0 is mirrored, so a
    //! wider request would have every beam read beam 0's pages.
    int addKvCache(KvCache& kvCache, int row = -1);

    //! Free the cache's row, leaving a hole. Throws when it is not a member.
    void removeKvCache(KvCache& kvCache);

    //! Resize every member, then publish once.
    //!
    //! The sanctioned way to advance a batched step, and the reason it exists: it waits on
    //! the previous publish before touching anyone, so the staging hazard is closed by
    //! construction rather than by a caller remembering waitPublished(). Per-request
    //! operations with no Batch equivalent -- commit(), a beam fork, an SWA recycle -- still
    //! need that call, so the simple rule is to make it once at the top of a step.
    //!
    //! One result per row, in row order; false for a free row. A member that fails rolls
    //! itself back, so a batch can partially fail and the table published afterwards
    //! describes the post-rollback state.
    //!
    //! `capacities` and `historyLengths` are indexed by row, so both must be maxBatchSize
    //! long. A nullopt leaves that dimension alone, matching KvCache::resize.
    std::vector<bool> resize(std::vector<std::optional<int>> const& capacities,
        std::vector<std::optional<int>> const& historyLengths, CUstream stream);

    //! Copy the staged rows to the device, enqueuing on `stream`.
    //!
    //! The staging rows are written by the members themselves, so they are always current;
    //! what this adds is a stream-ordered snapshot. Without it the host would be mutating
    //! the table a previously enqueued fetch is still reading. That snapshot is why a
    //! borrowed table is still copied rather than read in place: the host has to be free to
    //! write the next step's rows long before the kernels reading this step's have retired.
    //!
    //! Borrowing costs one strided transfer per layer group instead of one linear transfer
    //! for all of them, because the lender's rows are further apart than they are wide.
    //!
    //! Two stages. The host-to-device fill runs on a stream of this Batch's own, so it can
    //! overlap the previous step; `stream` then carries one linear device-to-device copy
    //! into the tables the fetch reads. Everything expensive is in the first stage, and
    //! only the second has to wait for anything.
    //!
    //! **The second stage must be ordered after the kernels that read the previous
    //! publish.** Those live in a captured graph, so the tables cannot be double-buffered
    //! out of the hazard -- the capture baked their addresses, which is why the staging
    //! sits upstream of them instead. Issuing this on the stream the graph replays on is
    //! what provides that ordering today. A torn publish is not a stale answer: the cache
    //! views carry pointers, so it can pair one request's residency with another's lease.
    //!
    //! The copy is asynchronous and reads pinned memory, so it is still in flight when this
    //! returns. **Every later host write to a member's page indices must be separated from
    //! it by waitPublished()** -- a resize(), a commit(), a beam fork, an SWA recycle,
    //! anything that locks or unlocks a page. Writing sooner races the copy and publishes a
    //! torn table.
    //!
    //! Throws when the previous publish has not been waited on, which is how that rule is
    //! enforced: this call stages the watermarks and views itself, so it would be writing
    //! into the same hazard. It refuses rather than waiting because a wait here covers only
    //! its own write -- a member mutated before the call has already raced by then, and
    //! waiting would hide it.
    void publish(CUstream stream);

    //! Block until the last publish() has finished reading the staging rows.
    //!
    //! Cheap once complete, and a no-op when nothing is in flight, so a caller may use it
    //! freely at the top of a step. Batch's own mutators call it; a caller manipulating a
    //! member directly is the one that has to.
    void waitPublished();

    [[nodiscard]] PageTableView pageTable(LayerGroupId layerGroupId) const;

    //! Per-row sparse cache state for one layer group, as the fetch kernel sees it.
    //!
    //! A device array of SparseCacheView, indexed by row. A free row, or a member whose
    //! cache is not reserved, reads as a default-constructed view -- null pointers and zero
    //! counts -- which the kernel skips.
    //!
    //! The views hold pointers into each request's own allocation, so what is mirrored is
    //! the handle rather than the memory: the reservation stays with the request. They move
    //! whenever a member resumes or suspends, which is why they are restaged on every
    //! publish alongside the watermarks.
    [[nodiscard]] CUdeviceptr sparseCacheViews(LayerGroupId layerGroupId) const;

    //! Per-row history length and capacity, published by the same transfer as the page
    //! tables. They have to travel together: the lengths bound which entries of a row are
    //! valid, so a kernel pairing one step's table with another step's lengths would read
    //! past the end of a row that has since been reused.
    [[nodiscard]] WatermarkView watermarks() const;

    //! Scratch for one layer group's fetch. Null pointers when the group carries no sparse
    //! cache, which is what makes "nothing to fetch here" cheap to detect.
    [[nodiscard]] SparseFetchScratch const& sparseFetchScratch(LayerGroupId layerGroupId) const;

    //! Row of a member, or -1.
    [[nodiscard]] int rowOf(KvCache const& kvCache) const;

    [[nodiscard]] int maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] int maxBlocksPerSeq() const noexcept
    {
        return mMaxBlocksPerSeq;
    }

    //! Members, by row; null where a row is free.
    [[nodiscard]] std::vector<KvCache*> const& rows() const noexcept
    {
        return mRows;
    }

private:
    friend class KvCache;

    //! Called by a member being destroyed, so its row does not dangle.
    void _detach(KvCache& kvCache) noexcept;

    //! Host-side page-index row for one (layer group, row): inside the pinned staging block
    //! when the table is this Batch's, inside the lender's when it is borrowed.
    [[nodiscard]] int32_t* _stagingRow(LayerGroupId layerGroupId, int row) const;

    //! Element strides of the host page-index rows, whoever owns them.
    [[nodiscard]] int64_t _pageTableRowStride() const noexcept;
    [[nodiscard]] int64_t _pageTableGroupStride() const noexcept;

    //! Bytes the device page table occupies ahead of the row state, when the staged block
    //! does not carry one. Zero otherwise, so every staging offset reaches its device
    //! counterpart by adding this and nothing else.
    [[nodiscard]] size_t _deviceBias() const noexcept;

    //! Bytes of a device block: the page table, the row state and the views.
    [[nodiscard]] size_t _deviceBytes() const noexcept;

    //! Block until every transfer this Batch issued has retired, so its buffers can go.
    void _drain() noexcept;

    [[nodiscard]] size_t _rowElements() const noexcept;

    //! Element offset of the watermark regions within the staged block.
    [[nodiscard]] size_t _historyOffset() const noexcept;
    [[nodiscard]] size_t _capacityOffset() const noexcept;
    [[nodiscard]] size_t _totalElements() const noexcept;

    //! Refresh the staged watermarks and cache views from the members. Cheap, and it keeps
    //! KvCache ignorant of the staging: only its page indices are written through
    //! setBasePageIndexBuf.
    void _stageRowState();

    void _allocateFetchScratch();

    //! Byte offset of the cache view region within the staged block.
    [[nodiscard]] size_t _viewsOffsetBytes() const noexcept;
    [[nodiscard]] size_t _totalBytes() const noexcept;

    KvCacheManager* mManager = nullptr;
    int mMaxBatchSize = 0;
    int mMaxBlocksPerSeq = 0;
    LifeCycleId mNumLayerGroups{0};

    std::vector<KvCache*> mRows;

    //! Set when the page-index table is borrowed, in which case mStaging and mDevice hold
    //! only the watermarks and cache views.
    std::optional<ExternalPageTable> mPageTable;

    //! Pinned so the members' index writes land here with no extra copy, and so the
    //! publish is a single asynchronous transfer per layer group.
    std::unique_ptr<HostMem> mStaging;

    //! What the fetch reads. Its address is baked into a captured graph, so it is written
    //! only on the caller's stream, after the replay that read the previous publish.
    CudaUniqPtr mDevice;

    //! A device-side copy of the same layout, filled from the host ahead of time.
    //!
    //! Nothing captured reads it, so its fill is free to overlap the previous step --
    //! which is the only way the host-to-device crossing can overlap compute at all. With
    //! one buffer the crossing has to follow the previous replay, since the kernels
    //! reading the table are inside it and the table's address is fixed by capture.
    //! Double-buffering the table itself cannot work for that same reason, so the second
    //! buffer goes upstream of the fixed address instead.
    CudaUniqPtr mDeviceStaging;

    //! Carries the host-to-device fill, off the caller's stream so it can run early.
    CachedCudaStream mUploadStream;

    //! One allocation holding every layer group's fetch scratch, carved per group.
    CudaUniqPtr mFetchScratch;
    TypedVec<LifeCycleId, SparseFetchScratch> mScratch;

    //! Recorded after the host-to-device fill, so the host can tell when the staged rows
    //! are free to write again. Null when no publish is in flight.
    CachedCudaEvent mPublished = CachedCudaEvent::makeNull();

    //! Recorded after the device staging has been read into the final tables. The next
    //! fill must wait on it, since that fill runs on another stream and would otherwise
    //! overwrite the staging while this step's copy is still reading it.
    CachedCudaEvent mStagingRead = CachedCudaEvent::makeNull();
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
