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

#include "kv_cache_manager_v2/sparseCache.h"

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Written to an output slot that carries no selection, i.e. selection padding.
inline constexpr int32_t kNoSparseSlot = -1;

//! Status codes a resolve launch can report back through SparseResolveArgs::status.
enum class SparseResolveStatus : int32_t
{
    kOk = 0,
    //! More misses than evictable entries. Reservation is supposed to make this impossible;
    //! it means the cache was sized below the selection width plus the in-flight span.
    kOutOfEntries = 1,
};

//! One layer group's resolve: turn a selection of unit ordinals into cache entry indices,
//! allocating entries for the units that are not resident.
//!
//! Layouts are [layer][...] row-major, matching SparseCacheView. Every array is device memory.
struct SparseResolveArgs
{
    //! Resident set, recency order and shape. Updated in place.
    SparseCacheView cache;

    //! [numLayers][selectionWidth] unit ordinals, negative where the selection is padded.
    //! Ordinals must be unique within a layer; duplicates would be resolved as separate
    //! misses and waste an entry each.
    int32_t const* selected = nullptr;
    //! Selected units per layer, uniform across layers and requests so shapes stay static.
    int32_t selectionWidth = 0;

    //! Unit ordinals at or above this are in flight: they hold the only copy of their data,
    //! so they are pinned against eviction. Equal to history_length / unit_tokens; see
    //! "Unflushed entries carry no state" in SPARSE_ATTENTION_DESIGN.md.
    int32_t historyUnits = 0;

    //! [numLayers][selectionWidth] entry index per selection, kNoSparseSlot where padded.
    int32_t* outSlot = nullptr;

    //! [numLayers][selectionWidth] the units that were not resident, and the entries given
    //! to them. The copy stage reads these; only the first missCount[layer] are valid.
    int32_t* missOrdinal = nullptr;
    int32_t* missSlot = nullptr;
    //! [numLayers] number of misses.
    int32_t* missCount = nullptr;

    //! [1] SparseResolveStatus. The caller must clear it before the launch.
    int32_t* status = nullptr;
};

//! Shared memory one CTA needs for `entriesPerLayer` entries and `selectionWidth` selections.
//! Grows as next_pow2(2 * entriesPerLayer) for the probe index, so it is the quantity that
//! bounds how large a cache a single launch can serve.
[[nodiscard]] size_t sparseResolveSharedBytes(int32_t entriesPerLayer, int32_t selectionWidth);

//! Largest entriesPerLayer this kernel can serve on `device`, given its shared memory limit.
[[nodiscard]] int32_t sparseResolveMaxEntries(int32_t selectionWidth, CUdevice device);

//! Launch the resolve. One CTA per layer; enqueues and returns.
//!
//! Throws std::invalid_argument when the shape is inconsistent or needs more shared memory
//! than the device allows.
void launchSparseResolve(SparseResolveArgs const& args, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
