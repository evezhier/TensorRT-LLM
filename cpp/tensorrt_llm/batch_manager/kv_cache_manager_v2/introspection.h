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
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

class KvCacheIntrospection
{
public:
    using ActivePageStats = std::tuple<TypedVec<CacheLevel, int>, TypedVec<CacheLevel, int>>;

    static ActivePageStats activePageStats(KvCache const& kvCache);

    // Whether the sequence's page at (ordinal, lcId) still points at a tree block;
    // nullopt when the slot is empty or holds an uncommitted page. Test hook for the
    // back-pointer invariant Block::replacePage() maintains.
    static std::optional<bool> committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId);
    static bool allTreePagesDroppable(KvCacheManager& manager);

    // White-box hooks: the device-resident state of a request's sparse history cache for
    // one life cycle. Synchronises and copies back, so it is test-only. Empty when the
    // life cycle has no cache reserved.
    static std::vector<int32_t> sparseCacheResidentOrdinals(KvCache const& kvCache, int lcId);

    // White-box hook: run the sparse resolve kernel over caller-supplied state, so the
    // probe index, victim choice and recency rebuild can be exercised without a request.
    // Synchronises. Inputs are [layer][entry] and [layer][selectionWidth] row-major.
    struct SparseResolveProbe
    {
        std::vector<int32_t> residentOrdinal; //!< [layer][entry] after the resolve
        std::vector<int32_t> lruOrder;        //!< [layer][entry] after the resolve
        std::vector<int32_t> outSlot;         //!< [layer][width] entry per selection
        std::vector<int32_t> missOrdinal;     //!< [layer][width], first missCount valid
        std::vector<int32_t> missSlot;        //!< [layer][width], first missCount valid
        std::vector<int32_t> missCount;       //!< [layer]
        std::vector<int32_t> inFlightSlot;    //!< [layer][maxInFlightUnits]
        int32_t status = 0;                   //!< SparseResolveStatus
    };

    static SparseResolveProbe sparseResolve(std::vector<int32_t> const& residentOrdinal,
        std::vector<int32_t> const& lruOrder, int numLayers, int entriesPerLayer, std::vector<int32_t> const& selected,
        int selectionWidth, int historyUnits, int capacityUnits, int maxInFlightUnits);

    // White-box hook: run the sparse unit copy over caller-supplied pools, so the address
    // arithmetic can be checked against a known byte pattern. Returns the destination pool
    // after the copy. Synchronises.
    static std::vector<uint8_t> sparseUnitCopy(std::vector<uint8_t> const& hostPool, uint64_t hostPageStride,
        std::vector<int32_t> const& pageTable, std::vector<uint8_t> const& devicePool, uint64_t devicePageStride,
        std::vector<int32_t> const& cachePages, std::vector<uint32_t> const& srcBufferOffset,
        std::vector<uint32_t> const& dstBufferOffset, uint32_t numChunks, uint32_t chunkBytes, uint32_t chunkStride,
        uint32_t unitStride, int expansion, std::vector<int32_t> const& missOrdinal,
        std::vector<int32_t> const& missSlot, std::vector<int32_t> const& missCount, int selectionWidth, int numLayers,
        bool writeBack);

    // White-box hook: the entries holding units a watermark move has just passed.
    // Synchronises.
    struct SparseFlushList
    {
        std::vector<int32_t> ordinal; //!< [layer][maxFlushUnits]
        std::vector<int32_t> slot;    //!< [layer][maxFlushUnits]
        std::vector<int32_t> count;   //!< [layer]
        int32_t status = 0;
    };

    static SparseFlushList sparseFlushList(std::vector<int32_t> const& residentOrdinal, int numLayers,
        int entriesPerLayer, int fromUnits, int toUnits, int maxFlushUnits);

    // White-box hook: minimum per-pool-group slot counts to support a BatchDesc.
    // Reaches StorageManager::computePoolGroupSlotsForBatch() (private) via friendship.
    static TypedVec<PoolGroupIndex, SlotCount> computeSlotsForBatch(KvCacheManager& manager, BatchDesc const& batch,
        int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse);

    // White-box test hooks: mutate auto-tuner state so accuracy tests can force a
    // pool rebalance. Reach KvCacheManager's private members via friendship.
    static void setNumSampledKvCaches(KvCacheManager& manager, int value);
    static void setLastAdjustmentTime(KvCacheManager& manager, double value);
    static void setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value);
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
