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

#include <cuda.h>

#include <cstdint>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! One converted index table, as the conversion sees it.
//!
//! A table holds `numEntries` values per (row, block): one per buffer of the requests it
//! serves, differing only by the offset baked into them. Merged requests share these
//! entries and are told apart by their base pointers, which the conversion never sees.
struct IndexTableDesc
{
    //! Raw slot ids for this table's life cycle: [numRows][rawRowStride].
    int32_t const* raw = nullptr;
    //! Destination: [numEntries][numRows][blocksPerRow].
    int32_t* converted = nullptr;
    //! Offset added to each entry, in page-index units. Length numEntries.
    int32_t const* offsets = nullptr;
    int32_t numEntries = 0;
    //! Slot id to page index, i.e. how many buffers the pool coalesces.
    int32_t scale = 1;

    //! Scratch metadata for this table's life cycle, null when it cannot hold scratch.
    //!
    //! A scratch block's slot does not come from the page table at all: it comes from the
    //! request's own scratch slots, indexed by how far into the scratch range the block
    //! sits. Null `scratchBegs` means the plain path applies to every block.
    int32_t const* scratchBegs = nullptr;  //!< [numRows]
    int32_t const* scratchEnds = nullptr;  //!< [numRows]
    int32_t const* scratchSlots = nullptr; //!< [numRows][scratchSlotStride]
    int32_t scratchSlotStride = 0;
    //! Sub-pages a scratch block occupies, which is what advances the slot cursor.
    int32_t scratchPagesPerBlock = 1;
};

//! Convert raw slot ids into kernel page indices for every table at once.
//!
//! One launch rather than one per table: a PER_LAYER mode gives a table per layer, and the
//! per-launch driver cost would then dominate what is a few arithmetic operations per
//! entry.
//!
//! `entry = slot * scale + offset`, with kBadPageIndex passed through unchanged -- a sparse
//! table numbers pure-history and input-token entries in different levels' pools, so only
//! the sentinel separates an absent block from slot zero.
//!
//! @param descs        device array of numTables descriptors
//! @param maxEntries    widest table's numEntries; narrower tables return immediately,
//!                      which costs less than giving each its own launch
//! @param rawRowStride  elements between consecutive rows of a raw table
void launchIndexConversion(IndexTableDesc const* descs, int32_t numTables, int32_t maxEntries, int32_t numRows,
    int32_t blocksPerRow, int32_t rawRowStride, CUstream stream);

//! The same, for tables whose life cycle substitutes scratch slots.
//!
//! Separate rather than a branch in the kernel above: a scratch block takes its slot from
//! the request instead of the page table, and wraps the layer's offset within the slot
//! rather than adding it, so the two paths share only their shape.
//!
//! Scratch is a PER_LAYER-only concern. Under SHARED every layer of a life cycle reads one
//! table, which substitution would break.
void launchIndexConversionWithScratch(IndexTableDesc const* descs, int32_t numTables, int32_t maxEntries,
    int32_t numRows, int32_t blocksPerRow, int32_t rawRowStride, CUstream stream);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
