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
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/utils/cudaEvent.h"
#include "kv_cache_manager_v2/utils/hostMem.h"

#include <cuda.h>

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCache;
class KvCacheManager;

//! One layer group's page table as the device sees it. Always packed: the Batch allocates
//! it, so a row is exactly `blocksPerRow` entries.
struct GpuCoalescedPageTable
{
    CUdeviceptr data = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! One index table's converted page indices as the device sees them, laid out
//! [entries][numRows][blocksPerRow]. A buffer reads the entry `index_table_slot` gives it.
//!
//! Packed, like the tables either side of it: the Batch allocates it, so there is no stride
//! a caller has to be told separately.
struct ConvertedIndexTable
{
    CUdeviceptr data = 0;
    int32_t entries = 0;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! One layer group's page table as the host writes it, in the Batch's pinned staging.
//! Packed like the device one, and for the same reason: the Batch allocates it.
//!
//! A consumer that needs these indices for its own purposes reads this rather than lending
//! the Batch storage of its own. One buffer with one writer is what keeps a member's
//! binding alive for the whole decode; which side allocates it never mattered to that.
struct CoalescedPageTable
{
    int32_t* data = nullptr;
    int32_t numRows = 0;
    int32_t blocksPerRow = 0;
};

//! Identifies one converted index table of a Batch.
using IndexTableId = StrongIndex<int, struct IndexTableIdTag>;

//! Device scratch one layer group's step works through: what a fetch resolves into before
//! the copy and the index scatter read it back, and what a write-back lists before its own
//! copy reads it.
//!
//! Batch-owned rather than per request, because both consumers are one launch spanning
//! every row and a kernel takes one pointer. The reservation in a request's lease is its
//! own; this is the step's.
//!
//! Owned by Batch rather than allocated per call because every extent is a configuration
//! constant -- rows, selection width, in-flight headroom -- so one allocation serves every
//! step, and a step that allocates nothing is a step a CUDA graph can capture.
struct SparseStepScratch
{
    int32_t* outSlot = nullptr;      //!< [numRows][selectionWidth]
    int32_t* missCount = nullptr;    //!< [numRows]
    int32_t* inFlightSlot = nullptr; //!< [numRows][maxInFlightUnits]
    int32_t selectionWidth = 0;
    int32_t maxInFlightUnits = 0;

    //! The misses of one layer, compacted across rows. A fetch resolves one layer at a
    //! time, so a single list serves the whole batch.
    SparseWorkList fetchWork;

    //! The write-back's counterpart. A write-back crosses every layer of a row at once and
    //! the copy that consumes it walks buffers, so it takes one list per layer rather than
    //! the one the fetch folds away.
    SparseWorkList flushWork;
    int32_t* flushCount = nullptr; //!< [numRows][numLayers]
    int32_t numLayers = 0;

    //! What differs between the pools of a copy, on the device for the kernel to index by
    //! `blockIdx.z`. Built and uploaded once with the Batch: every field comes from the
    //! buffer plan and the index tables, both fixed at construction, so a captured graph
    //! replays a copy with no host write at all.
    //!
    //! [numBuffers] one copy descriptor per BufferId, serving both directions. BufferIds
    //! are ordered (layer, pool), so one already names its own layer's index-table entry
    //! and needs no layer dimension; the fetch and the write-back then want the same
    //! descriptors and differ only in which of them they launch over.
    SparseCopyDesc const* copyDesc = nullptr;
    int32_t numBuffers = 0;
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
    //! @param mode page index mode, applied to every life cycle. Left unset, each life
    //! cycle takes the smaller table -- SHARED where the manager says it is always
    //! available, PER_LAYER otherwise.
    //!
    //! The mode is fixed here because a converted table's shape is baked into a captured
    //! graph. It cannot follow a member's scratch state, which varies per step.
    //! @param indexTables what consumers need addressable together. Requests whose
    //! converted indices coincide are merged onto one table, which is how a quantised
    //! layer's data and block scales come to share one.
    //! @param stream the stream every member runs on. A Batch owns it rather than taking
    //! one per call: a member's own work and the Batch's transfers have to be ordered
    //! against each other, and two streams would leave that ordering to the caller.
    //!
    //! @param maxTotalInputTokens tokens one step may ingest across every row, as the
    //! runtime's own batching bound gives it. It sizes the write-back's work list: a row
    //! advancing `t` tokens crosses at most `t / unitTokens + 1` units, so the batch
    //! crosses at most `maxTotalInputTokens / unitTokens + maxBatchSize` of them.
    //!
    //! Only an aggregate bound works here, and only because the list is compacted across
    //! rows. Were it indexed by row, each row's slice would have to allow for that row
    //! taking the whole budget, and the product would be hundreds of times larger.
    //!
    //! Left unset it falls back to `maxBatchSize * maxInFlightUnits`, which today is
    //! `maxBatchSize` -- a decode step advances capacity one token per row, so one row
    //! crosses at most one unit boundary. Pass it once a step can ingest more than that,
    //! which is what chunked prefill under in-flight batching does.
    explicit Batch(KvCacheManager& manager, CUstream stream, int maxBatchSize = 0, int maxBlocksPerSeq = 0,
        std::optional<PageIndexMode> mode = std::nullopt, std::vector<IndexTableRequest> indexTables = {},
        std::optional<int> maxTotalInputTokens = std::nullopt);

    //! The stream the Batch and every member run on, fixed at construction.
    [[nodiscard]] CUstream cudaStream() const noexcept
    {
        return mCudaStream;
    }

    //! Page index mode of one life cycle, as chosen at construction.
    [[nodiscard]] PageIndexMode indexMode(LayerGroupId layerGroupId) const;

    // ---- Converted index tables --------------------------------------------

    [[nodiscard]] IndexTableId numIndexTables() const noexcept;

    //! Every request one table satisfies. A request's buffer order is its own mapping into
    //! the buffer dimension, so the first buffer of each sits at entry zero.
    [[nodiscard]] std::vector<IndexTableRequest> const& indexTable(IndexTableId tableId) const;

    //! Which table serves this buffer, and its entry in that table's buffer dimension.
    //! This is the normal query: a kernel asks where its indices are and at what entry.
    [[nodiscard]] std::pair<IndexTableId, int> indexTableSlot(LayerId layerId, DataRole const& role) const;

    //! What a kernel adds to a converted index for this buffer. Every buffer of a request
    //! shares it, because the request's own offsets are already in the entries.
    [[nodiscard]] MemAddress indexTableBase(LayerId layerId, DataRole const& role) const;

    //! Where one table's converted indices live: int32[entries][max_batch][blocks].
    //! Stable for the Batch's lifetime, so a captured graph may bake it.
    [[nodiscard]] CUdeviceptr indexTableAddress(IndexTableId tableId) const;

    //! One index table's converted indices, shaped. The same memory indexTableAddress
    //! returns, with the extents a reader would otherwise have to assemble from
    //! numIndexTables(), index_table_slot() and the Batch's own bounds.
    [[nodiscard]] ConvertedIndexTable convertedIndexTable(IndexTableId tableId) const;

    //! How a sparse copy addresses a run of buffers that begins at `first`.
    //!
    //! The run's buffers occupy consecutive entries of one request, which construction
    //! checked, so naming the first is enough: the rest follow by entry.
    struct SparseIndexRun
    {
        //! The run's first entry, for row zero.
        CUdeviceptr table = 0;
        //! What a converted index is added to. Shared by the whole run.
        MemAddress base = 0;
        //! Elements between consecutive entries, and between consecutive rows.
        int entryStride = 0;
        int rowStride = 0;
    };

    [[nodiscard]] SparseIndexRun sparseIndexRun(LayerId layerId, DataRole const& role) const;
    ~Batch();

    Batch(Batch const&) = delete;
    Batch& operator=(Batch const&) = delete;

    //! Seat a cache in `row`, or in the lowest free row when it is negative.
    //!
    //!
    //! Throws when the row is taken or out of range, the batch is full, the cache already
    //! belongs to a batch, or its beam width exceeds one -- only beam 0 is mirrored, so a
    //! wider request would have every beam read beam 0's pages.
    int addKvCache(KvCache& kvCache, int row = -1);

    //! Free the cache's row, leaving a hole, and hand back the reference the Batch held.
    //! Throws when it is not a member.
    //!
    //! Returns the arena lease, which discards whatever the GPU cache held: a lease comes back
    //! with every entry free. Call flushSparseUnits() first for anything worth keeping.
    std::shared_ptr<KvCache> removeKvCache(KvCache& kvCache);

    //! The same, by the row addKvCache returned. Null when nothing is seated there, since
    //! a caller sweeping rows should not have to ask twice about each one.
    std::shared_ptr<KvCache> removeKvCache(int row);

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
    //! Rejects, before touching any member, a history length that falls or that outruns the
    //! capacity its row already had: capacity granted by this call makes room for tokens
    //! written after it, so history can only cover tokens whose space was already there.
    //!
    //! A row whose growth fails still takes its history advance, which is legal by that same
    //! check and needs no slots -- both the stale range and the scratch range are monotone
    //! in history, so raising it only releases. Its result is still 0: the row is not at the
    //! state that was asked for.
    std::vector<uint8_t> resize(
        std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths);

    //! Copy the staged rows to the device, enqueuing on the Batch's stream.
    //!
    //! The staging rows are written by the members themselves, so they are always current;
    //! what this adds is a stream-ordered snapshot. Without it the host would be mutating
    //! the table a previously enqueued fetch is still reading. That snapshot is why the
    //! table is copied rather than read in place: the host has to be free to write the next
    //! step's rows long before the kernels reading this step's have retired.
    //!
    //! Two stages. The host-to-device fill runs on an upload stream of this Batch's own, so
    //! it can overlap the previous step; the Batch's stream then carries one linear
    //! device-to-device copy into the tables the fetch reads. Everything expensive is in the
    //! first stage, and only the second has to wait for anything.
    //!
    //! **The second stage must be ordered after the kernels that read the previous
    //! publish.** Those live in a captured graph, so the tables cannot be double-buffered
    //! out of the hazard -- the capture baked their addresses, which is why the staging
    //! sits upstream of them instead. The Batch's stream being the one the graph replays on
    //! is what provides that ordering. A torn publish is not a stale answer: the cache
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
    void publish();

    //! Write back the units each row's watermark is about to pass, given the history the
    //! caller is about to set. One per row; a row may repeat its current history, which
    //! asks for nothing.
    //!
    //! **Call this before advancing history, not after.** A block must be in its host page
    //! before it can be selected, and selectable means below the watermark; resize() refuses
    //! a history this has not covered.
    //!
    //! Bounded by the capacity each row started from, not by how far history travels: only
    //! a unit that was in flight has a cell holding its one copy, so a resumed request
    //! moving history across a whole prompt writes back nothing. A move wider than the
    //! in-flight headroom is refused, because the per-row flush list is sized for one.
    //!
    //! Reads the index table as of the last publish, which covers the blocks in range for
    //! the same reason -- they were allocated by an earlier step.
    void flushSparseUnits(std::vector<int> const& newHistoryLengths);

    //! The history each row has been written back to. A row that has never needed a flush
    //! reads its own history, so a caller may always compare against it.
    [[nodiscard]] int flushedHistory(int row) const;

    //! Where one life cycle's staged [from] units start; its [to] units follow maxBatchSize
    //! later. Written by flushSparseUnits(), read by the scan it launches.
    [[nodiscard]] int32_t const* flushRangeRow(LifeCycleId lifeCycle) const;

    //! Block until the last publish() has finished reading the staging rows.
    //!
    //! Cheap once complete, and a no-op when nothing is in flight, so a caller may use it
    //! freely at the top of a step. Batch's own mutators call it; a caller manipulating a
    //! member directly is the one that has to.
    void waitPublished();

    [[nodiscard]] GpuCoalescedPageTable gpuCoalescedPageTable(LayerGroupId layerGroupId) const;

    //! One layer group's page table on the host, as a publish will upload it.
    [[nodiscard]] CoalescedPageTable coalescedPageTable(LayerGroupId layerGroupId) const;

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
    [[nodiscard]] SparseStepScratch const& sparseStepScratch(LayerGroupId layerGroupId) const;

    //! Row of a member, or -1.
    [[nodiscard]] int rowOf(KvCache const& kvCache) const;

    //! Members seated, which is the rows occupied.
    [[nodiscard]] int numKvCaches() const noexcept;

    //! Whether the occupied rows are exactly [0, numKvCaches()).
    //!
    //! A Batch does not require it. Removing a member leaves its row free and the next join
    //! takes the lowest free one, so a hole is legal and transient. An engine that maps rows
    //! onto its own request slots usually does require it, and asking here beats keeping a
    //! mirror of the membership to check against -- a mirror can disagree, this cannot.
    [[nodiscard]] bool isContiguous() const noexcept;

    [[nodiscard]] int maxBatchSize() const noexcept
    {
        return mMaxBatchSize;
    }

    [[nodiscard]] int maxBlocksPerSeq() const noexcept
    {
        return mMaxBlocksPerSeq;
    }

    //! Members, by row; null where a row is free.
    [[nodiscard]] std::vector<std::shared_ptr<KvCache>> const& rows() const noexcept
    {
        return mRows;
    }

private:
    friend class KvCache;

    //! Called by a member being destroyed. Unreachable while seated -- the Batch holds a
    //! reference, so the destructor cannot run -- and kept as the assertion of that.
    void _detach(KvCache& kvCache) noexcept;

    //! Host-side page-index row for one (layer group, row), inside the pinned staging block.
    [[nodiscard]] int32_t* _stagingRow(LayerGroupId layerGroupId, int row) const;

    //! Bytes of a device block: the page table, the row state and the views.
    [[nodiscard]] size_t _deviceBytes() const noexcept;

    //! Block until every transfer this Batch issued has retired, so its buffers can go.
    void _drain() noexcept;

    [[nodiscard]] size_t _rowElements() const noexcept;

    //! Element offset of the watermark regions within the staged block.
    [[nodiscard]] size_t _historyOffset() const noexcept;
    [[nodiscard]] size_t _capacityOffset() const noexcept;
    [[nodiscard]] size_t _totalElements() const noexcept;

    //! Seed a row's staged watermarks, so a row reads as its occupant rather than as
    //! whatever the last one left.
    void _seedRowWatermarks(int row, int historyLength, int capacity);

    //! Refresh the staged watermarks and cache views from the members. Cheap, and it keeps
    //! KvCache ignorant of the staging: only its page indices are written through
    //! setBasePageIndexBuf.
    void _stageRowState();

    void _allocateStepScratch();

    //! Entries one layer's write-back list must hold, from the step's token budget.
    [[nodiscard]] int32_t _flushCapacity(LifeCycleId lifeCycle) const;

    //! Fill the per-buffer copy descriptors and upload them. Runs once, after the index
    //! tables exist, because a descriptor names where in them a buffer's indices sit.
    void _allocateCopyDesc();

    //! Device room for the per-row unit ranges a write-back reads:
    //! [life cycle][2][maxBatchSize], as [from][to]. Allocated with the fetch scratch
    //! because it has the same lifetime and the same reason to exist -- a step that
    //! allocates nothing is a step a graph can capture.
    void _allocateFlushRanges();

    CudaUniqPtr<int32_t> mFlushRanges;
    //! Staged host-side, then handed to a copy that consumes it before returning.
    std::vector<int32_t> mFlushRangeStaging;
    //! Per row, the history its units have been written back to.
    std::vector<int> mFlushedHistory;
    //! Recorded after the write-back, so the next call's staging can wait for its reads.
    CachedCudaEvent mFlushRead = CachedCudaEvent::makeNull();

    //! Byte offset of the cache view region within the staged block.
    [[nodiscard]] size_t _viewsOffsetBytes() const noexcept;
    [[nodiscard]] size_t _totalBytes() const noexcept;

    KvCacheManager* mManager = nullptr;
    CUstream mCudaStream = nullptr;
    int mMaxBatchSize = 0;
    int mMaxBlocksPerSeq = 0;
    std::optional<int> mMaxTotalInputTokens;
    LifeCycleId mNumLayerGroups{0};

    //! Owning: a seated member cannot be destroyed under the Batch, so a caller is free to
    //! keep only the Batch. Leaving returns the reference. KvCache points back with a raw
    //! `mBatch`, so there is no cycle.
    std::vector<std::shared_ptr<KvCache>> mRows;

    //! Device storage behind every life cycle's per-BufferId copy descriptors.
    CudaUniqPtr<SparseCopyDesc> mCopyDesc;

    //! Every life cycle's work-list counters. A step clears them once, in publish(); a
    //! producer then raises its own and the copy that consumes the list reads it.
    //!
    //! Not owning: they lead `mStepScratch`, which is what makes one clear reach all of
    //! them, so that allocation frees them.
    int32_t* mCounters = nullptr;
    size_t mCounterElements = 0;

    //! Per life cycle, fixed at construction. Only scratch-slot substitution can make one
    //! layer's indices differ from another's in the same life cycle, so a life cycle that
    //! cannot hold scratch takes SHARED and the single smaller table that implies.
    TypedVec<LifeCycleId, PageIndexMode> mIndexModes;

    //! One converted index table: the requests it serves, and what a consumer needs to read
    //! it. `concatOffsets` is in page-index units, parallel to a request's buffers.
    struct IndexTable
    {
        std::vector<IndexTableRequest> served;
        LifeCycleId lifeCycle{0};
        int scale = 1;
        std::vector<int> concatOffsets;
        //! Element offset into mIndexTableMemory.
        size_t elementOffset = 0;
    };

    std::vector<IndexTable> mIndexTables;

    //! Where one buffer's indices are. The base belongs to the buffer's request, not to the
    //! table: merged requests share entries precisely so they can be read through different
    //! bases, which is what keeps a quantised layer's data and block scales apart.
    struct BufferSlot
    {
        IndexTableId table{0};
        int entry = 0;
        MemAddress base = 0;
    };

    std::map<std::pair<LayerId, DataRole>, BufferSlot> mBufferSlots;

    //! Validate the requests, merge those whose converted indices coincide, and record
    //! where each buffer landed.
    void _buildIndexTables(std::vector<IndexTableRequest> const& requests);

    //! Refuse a sparse life cycle whose buffers no single launch can address.
    //!
    //! A sparse copy walks one (layer, pool) run of buffers per launch, from one base
    //! pointer, so that run has to sit at consecutive entries of one request. Splitting it
    //! across requests would need two bases in one launch; declaring only part of it would
    //! leave the rest with no converted indices at all. Checked here because a run that
    //! cannot be addressed produces wrong page addresses at step time rather than an error.
    void _checkSparseRunsAreAddressable() const;

    //! Allocate the tables and the device-side descriptors the conversion reads.
    void _allocateIndexTables();

    //! Collect each member's scratch slots for the life cycles that can hold them.
    //! Kept out of the staged block because that block's offsets are shared with the
    //! watermarks and the views; a separate buffer leaves those alone for one small extra
    //! transfer on the upload stream.
    void _stageScratch();

    //! Elements of the scratch staging: [begs][ends][slots].
    [[nodiscard]] size_t _scratchElements() const noexcept;

    //! Host and device sides of the scratch metadata, laid out [begs][ends][slots].
    //! Empty when no table's life cycle can hold scratch.
    std::unique_ptr<HostMem> mScratchStaging;
    CudaUniqPtr<int32_t> mScratchDevice;
    //! Slot ids a request may hold for one life cycle, which bounds the staging.
    int mMaxScratchSlots = 0;
    //! Tables whose life cycle can hold scratch sort last, so one launch covers each kind.
    int mNumPlainTables = 0;

    //! Convert this step's raw slot ids into every table, on `stream`.
    void _convertIndices(CUstream stream);

    //! All converted tables, carved by IndexTable::elementOffset.
    CudaUniqPtr<int32_t> mIndexTableMemory;
    //! One IndexTableDesc per table, and the offsets they point at.
    CudaUniqPtr<> mIndexTableDescs;
    int mMaxTableEntries = 0;

    //! Pinned so the members' index writes land here with no extra copy, and so the
    //! publish is a single asynchronous transfer per layer group.
    std::unique_ptr<HostMem> mStaging;

    //! What the fetch reads. Its address is baked into a captured graph, so it is written
    //! only on the caller's stream, after the replay that read the previous publish.
    CudaUniqPtr<> mDevice;

    //! A device-side copy of the same layout, filled from the host ahead of time.
    //!
    //! Nothing captured reads it, so its fill is free to overlap the previous step --
    //! which is the only way the host-to-device crossing can overlap compute at all. With
    //! one buffer the crossing has to follow the previous replay, since the kernels
    //! reading the table are inside it and the table's address is fixed by capture.
    //! Double-buffering the table itself cannot work for that same reason, so the second
    //! buffer goes upstream of the fixed address instead.
    CudaUniqPtr<> mDeviceStaging;

    //! Carries the host-to-device fill, off the caller's stream so it can run early.
    CachedCudaStream mUploadStream;

    //! One allocation holding every layer group's fetch scratch, carved per group.
    CudaUniqPtr<int32_t> mStepScratch;
    TypedVec<LifeCycleId, SparseStepScratch> mScratch;

    //! Recorded after the host-to-device fill, so the host can tell when the staged rows
    //! are free to write again. Null when no publish is in flight.
    CachedCudaEvent mPublished = CachedCudaEvent::makeNull();

    //! Recorded after the device staging has been read into the final tables. The next
    //! fill must wait on it, since that fill runs on another stream and would otherwise
    //! overwrite the staging while this step's copy is still reading it.
    CachedCudaEvent mStagingRead = CachedCudaEvent::makeNull();
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
