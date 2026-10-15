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

#include "tensorrt_llm/common/assert.h"
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Cache tier configuration structs (mirrors _config.py)
// ---------------------------------------------------------------------------

struct GpuCacheTierConfig
{
    size_t quota = 0; // bytes

    CacheTier tier() const noexcept
    {
        return CacheTier::GPU_MEM;
    }

    void assertValid() const
    {
        if (quota == 0)
            throw std::invalid_argument("GpuCacheTierConfig: quota must be > 0");
    }
};

struct HostCacheTierConfig
{
    size_t quota = 0; // bytes

    CacheTier tier() const noexcept
    {
        return CacheTier::HOST_MEM;
    }

    void assertValid() const
    {
        if (quota == 0)
            throw std::invalid_argument("HostCacheTierConfig: quota must be > 0");
    }
};

struct DiskCacheTierConfig
{
    size_t quota = 0; // bytes
    std::string path; // directory for temp files

    CacheTier tier() const noexcept
    {
        return CacheTier::DISK;
    }

    void assertValid() const;
};

// Variant holding any tier config.
using CacheTierConfig = std::variant<GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig>;

// Helper to extract tier from a variant.
inline CacheTier cacheTierOf(CacheTierConfig const& cfg)
{
    return std::visit([](auto const& c) { return c.tier(); }, cfg);
}

inline size_t cacheTierQuota(CacheTierConfig const& cfg)
{
    return std::visit([](auto const& c) { return c.quota; }, cfg);
}

// ---------------------------------------------------------------------------
// Buffer configuration (one KV buffer inside an attention layer).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SelectionUnitLayout — where one selection unit's bytes sit inside a buffer.
//
// Buffer-relative, carrying no page offset: it describes the internal shape of the buffer,
// which is the part the model integration knows. The manager supplies the page offset from
// BufferAttr.
//
// Describes both sides of the copy, which have the same shape. The fetch reads hot pages in
// ordinary slot layout and writes cells holding one buffer's worth of the same structure, so
// one set of strides serves both with only the unit index differing. No codec is involved:
// a codec transforms pages leaving the hot tier, which the fetch never reads.
// ---------------------------------------------------------------------------
struct SelectionUnitLayout
{
    uint32_t numChunks = 1;   //!< One per head for a head-major page; 1 otherwise.
    uint32_t chunkBytes = 0;  //!< Contiguous bytes of one unit within one chunk.
    uint32_t chunkStride = 0; //!< Distance between consecutive chunks.
    uint32_t unitStride = 0;  //!< Distance between consecutive selection units.

    //! Derive the layout of a densely packed page.
    //!
    //! @param pageBytes One buffer's bytes for a whole page, i.e. BufferConfig::size.
    //! @param unitsPerPage Selection units the page holds.
    //! @param numChunks Pieces a unit is gathered from; see BufferConfig::sparseUnitChunks.
    static SelectionUnitLayout derive(size_t pageBytes, int unitsPerPage, int numChunks)
    {
        SelectionUnitLayout out;
        out.numChunks = static_cast<uint32_t>(numChunks);
        out.chunkStride = static_cast<uint32_t>(pageBytes / static_cast<size_t>(numChunks));
        out.chunkBytes = out.chunkStride / static_cast<uint32_t>(unitsPerPage);
        // Units pack densely inside a chunk, which is what makes the page's own extent
        // enough to place them: a unit's stride is its size.
        out.unitStride = out.chunkBytes;
        return out;
    }

    //! @param unitSize Bytes of one selection unit, i.e. size / unitsPerPage.
    void validate(size_t unitSize) const
    {
        if (numChunks == 0 || chunkBytes == 0)
        {
            throw std::invalid_argument("SelectionUnitLayout: numChunks and chunkBytes must be non-zero");
        }
        if (static_cast<size_t>(numChunks) * chunkBytes != unitSize)
        {
            throw std::invalid_argument("SelectionUnitLayout: numChunks * chunkBytes must equal one unit's bytes");
        }
        if (numChunks > 1 && chunkStride < chunkBytes)
        {
            throw std::invalid_argument("SelectionUnitLayout: chunkStride must not overlap consecutive chunks");
        }
        if (unitStride < chunkBytes)
        {
            throw std::invalid_argument("SelectionUnitLayout: unitStride must not overlap consecutive units");
        }
    }
};

struct BufferConfig
{
    DataRole role;
    size_t size = 0; // bytes per page

    // If set, overrides tokens_per_block for this buffer.
    // Must be a divisor of KVCacheManagerConfig::tokensPerBlock.
    std::optional<int> tokensPerBlockOverride;

    bool isSparse = false; //!< Whether history uses the sparse attention lifecycle.

    //! Pieces a selection unit is gathered from within its page. Sparse buffers only.
    //!
    //! Per buffer, because it is the dimension order of *this* buffer's page:
    //!
    //!   - **1** for a sequence-major page (`[token][head][dim]`), where a unit's tokens
    //!     carry all their heads with them and the whole unit is one contiguous range.
    //!     Also 1 for any single-head page -- a compressed or pooled latent, say -- where
    //!     the two orders are the same bytes.
    //!   - **numHeads** for a head-major page (`[head][token][dim]`), where each head
    //!     contributes its own contiguous run and the unit is that many strided pieces.
    //!
    //! Everything else about the layout follows from this, the layer's unit size and the
    //! page being densely packed, so a padded page -- an alignment gap between heads or
    //! between units -- is not expressible.
    int sparseUnitChunks = 1;

    void validate() const
    {
        if (!isSparse && sparseUnitChunks != 1)
        {
            throw std::invalid_argument("BufferConfig: sparseUnitChunks set on a non-sparse buffer");
        }
        if (isSparse && tokensPerBlockOverride.has_value())
        {
            // Both would name a unit count, and nothing makes them agree. The override is a
            // storage-wide reinterpretation; a sparse layer says what it means with
            // SparseCacheConfig::unitTokens, which leaves pages and index spaces alone.
            throw std::invalid_argument(
                "BufferConfig: tokensPerBlockOverride is not supported on a sparse buffer; use the layer's "
                "sparse_cache.unit_tokens");
        }
        if (sparseUnitChunks <= 0)
        {
            throw std::invalid_argument("BufferConfig: sparseUnitChunks must be positive");
        }
    }

    //! Where this buffer's selection units sit inside a page, given how many the layer
    //! divides a page into. Throws when the page cannot be divided as declared, which is a
    //! configuration error rather than a fetch-time one.
    [[nodiscard]] SelectionUnitLayout sparseLayout(int unitsPerPage) const
    {
        auto const pieces = static_cast<size_t>(sparseUnitChunks) * static_cast<size_t>(unitsPerPage);
        if (size == 0 || pieces == 0 || size % pieces != 0)
        {
            throw std::invalid_argument(
                "BufferConfig: size must divide evenly into sparseUnitChunks * units-per-page pieces");
        }
        return SelectionUnitLayout::derive(size, unitsPerPage, sparseUnitChunks);
    }
};

// ---------------------------------------------------------------------------
// SparseCacheConfig — one layer's sparse fetch path.
//
// Per layer rather than per buffer because every sparse buffer of a layer shares one
// resolve, one resident set and one recency order: the metadata is laid out [layer][entry],
// and a selection names units for all of them at once. Declaring these per buffer would let
// two of them disagree with nothing to reconcile it.
//
// Its presence is what configures the fetch path. A sparse buffer in a layer without it
// still gets sparse storage and locking; it simply has no GPU cache to read through.
// ---------------------------------------------------------------------------
struct SparseCacheConfig
{
    //! Selection width: units a sparse-attention step selects for this layer.
    int numSelectedBlocks = 0;

    //! Cells one buffer gets in a request's GPU cache, counting both populations: the
    //! history it caches and every unit still in flight.
    //!
    //! Two to four times numSelectedBlocks is the usual choice: a cache only as wide as the
    //! selection retains nothing between steps, so what it holds above the selection is
    //! what survives to be hit next step. Only one unit of it is ever in flight, because a
    //! sparse request contributes one query token per step, so the rest is all reuse.
    int maxCachedUnits = 0;

    //! Original tokens one selection unit spans. Unset means a unit is a whole page.
    //!
    //! Must divide KVCacheManagerConfig::tokensPerBlock; the quotient is how many units a
    //! page holds. Distinct from BufferConfig::tokensPerBlockOverride, which is a
    //! storage-wide reinterpretation of a buffer's page: this names the sparse cache's
    //! granularity and nothing else, so pages, page indices and index spaces all keep the
    //! manager's tokensPerBlock.
    std::optional<int> unitTokens;

    //! Selection units one page holds, given the manager's tokensPerBlock.
    [[nodiscard]] int unitsPerPage(int tokensPerBlock) const
    {
        return tokensPerBlock / unitTokens.value_or(tokensPerBlock);
    }

    void validate(int tokensPerBlock) const
    {
        if (numSelectedBlocks <= 0 || maxCachedUnits <= 0)
        {
            throw std::invalid_argument(
                "SparseCacheConfig: numSelectedBlocks and maxCachedUnits must be positive; omit sparse_cache "
                "entirely for a layer with no fetch path");
        }
        if (maxCachedUnits <= numSelectedBlocks)
        {
            // Equality leaves no cell for the unit being written, so the request could not
            // take a single token. The cache holds a full selection plus everything in
            // flight, and those populations are disjoint: selection is below the watermark,
            // in flight is above it.
            throw std::invalid_argument(
                "SparseCacheConfig: maxCachedUnits must exceed numSelectedBlocks, to leave room for units in "
                "flight");
        }
        if (unitTokens.has_value() && (*unitTokens <= 0 || tokensPerBlock % *unitTokens != 0))
        {
            throw std::invalid_argument("SparseCacheConfig: unitTokens must be a divisor of tokensPerBlock");
        }
    }
};

// ---------------------------------------------------------------------------
// Layer type discriminator.
// ---------------------------------------------------------------------------

enum class LayerType : int
{
    ATTENTION = 0,
    SSM = 1,
};

// ---------------------------------------------------------------------------
// Attention layer configuration.
// ---------------------------------------------------------------------------

namespace detail
{

inline void validateNoDuplicateBufferRoles(std::vector<BufferConfig> const& buffers)
{
    std::unordered_set<DataRole> roles;
    for (auto const& buf : buffers)
    {
        TLLM_CHECK(roles.insert(buf.role).second);
    }
}

} // namespace detail

struct AttentionLayerConfig
{
    static constexpr LayerType type = LayerType::ATTENTION;

    LayerId layerId = 0;
    std::vector<BufferConfig> buffers;

    // nullopt = no sliding window.
    std::optional<int> slidingWindowSize;

    // nullopt or 0 = no sink tokens.
    std::optional<int> numSinkTokens;

    //! The sparse fetch path for this layer's sparse buffers, if it has one. Its presence
    //! is what configures the GPU cache; sparse storage and locking work without it.
    std::optional<SparseCacheConfig> sparseCache;

    [[nodiscard]] std::optional<int> windowSize() const noexcept
    {
        return slidingWindowSize;
    }

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
        // A layer may mix sparse and dense buffers. Sparsity decides which pool a buffer
        // lands in, and nothing above that, so an indexer and the data it selects over can
        // stay in the one layer they belong to.
        for (auto const& buf : buffers)
        {
            buf.validate();
        }
        if (sparseCache.has_value()
            && std::none_of(buffers.begin(), buffers.end(), [](BufferConfig const& b) { return b.isSparse; }))
        {
            throw std::invalid_argument(
                "AttentionLayerConfig: sparse_cache is set on a layer with no sparse buffer to serve");
        }
    }
};

// ---------------------------------------------------------------------------
// SSM (State Space Model) layer configuration.
// ---------------------------------------------------------------------------

struct SsmLayerConfig
{
    static constexpr LayerType type = LayerType::SSM;

    LayerId layerId = 0;
    std::vector<BufferConfig> buffers;

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
        for (auto const& buf : buffers)
        {
            if (buf.tokensPerBlockOverride.has_value())
            {
                throw std::invalid_argument("tokensPerBlockOverride not supported for SSM layers");
            }
            if (buf.isSparse)
            {
                throw std::invalid_argument("Sparse buffers are only supported for attention layers");
            }
            buf.validate();
        }
    }
};

using LayerConfig = std::variant<AttentionLayerConfig, SsmLayerConfig>;

// ---------------------------------------------------------------------------
// KVCacheDesc — describes one KV cache request's capacity and history length.
// Mirrors _config.py::KVCacheDesc.
// ---------------------------------------------------------------------------
struct KVCacheDesc
{
    int capacity = 0;
    int historyLength = 0;
    // Beam search does not replicate the whole cache. Blocks that lie entirely
    // inside the prompt are committed and canonicalized to beam 0 (see
    // KvCache::_appendBeams(), which skips ordinals below promptLength /
    // tokensPerBlock and every committed block); only the prompt tail onward is
    // replicated. Sizing that scales everything by beamWidth would cancel out in
    // the normalized pool ratio and leave the skew between life cycles in place,
    // so the split has to be modelled explicitly.
    int beamWidth = 1;
    int promptLength = 0;

    void validate() const
    {
        TLLM_CHECK(0 <= historyLength && historyLength <= capacity);
        TLLM_CHECK(beamWidth >= 1);
        TLLM_CHECK(0 <= promptLength);
    }

    // Value equality, mirroring the Python @dataclass(frozen=True) semantics the
    // bindings replace. Required so tests can compare descs by value.
    bool operator==(KVCacheDesc const& other) const noexcept
    {
        return capacity == other.capacity && historyLength == other.historyLength && beamWidth == other.beamWidth
            && promptLength == other.promptLength;
    }

    bool operator!=(KVCacheDesc const& other) const noexcept
    {
        return !(*this == other);
    }
};

// ---------------------------------------------------------------------------
// BatchDesc — a batch of requests that the KVCacheManager must support.
// Mirrors _config.py::BatchDesc.
// ---------------------------------------------------------------------------
struct BatchDesc
{
    std::vector<KVCacheDesc> kvCaches;
    int systemPromptLength = 0; // tokens shared by all requests (0 if no reuse)

    void validate() const
    {
        TLLM_CHECK(systemPromptLength >= 0);
        for (auto const& desc : kvCaches)
        {
            desc.validate();
        }
    }

    // Value equality, mirroring the Python @dataclass(frozen=True) semantics the
    // bindings replace. Uses KVCacheDesc::operator== elementwise.
    bool operator==(BatchDesc const& other) const noexcept
    {
        return systemPromptLength == other.systemPromptLength && kvCaches == other.kvCaches;
    }

    bool operator!=(BatchDesc const& other) const noexcept
    {
        return !(*this == other);
    }
};

// ---------------------------------------------------------------------------
// SWA scratch reuse configuration.
// ---------------------------------------------------------------------------
struct SwaScratchReuseConfig
{
    int maxRewindLen = 0;

    void validate() const
    {
        if (maxRewindLen < 0)
        {
            throw std::invalid_argument("SwaScratchReuseConfig: max_rewind_len must be non-negative");
        }
    }
};

// ---------------------------------------------------------------------------
// Top-level KV cache manager configuration (mirrors _config.py::KVCacheManagerConfig).
// ---------------------------------------------------------------------------

struct KVCacheManagerConfig
{
    int tokensPerBlock = 0;

    // Ordered from warm (GPU) to cold (disk). First must be GPU memory.
    std::vector<CacheTierConfig> cacheTiers;

    // Layer configs (attention or SSM). Layer IDs must be unique.
    std::vector<LayerConfig> layers;

    // Suspend/resume threshold: if utilization > this, resuming will fail.
    float maxUtilForResume = 0.97f;

    // Try to reuse tokens from partially matched blocks.
    bool enablePartialReuse = true;

    // Tokens dropped from the tail of every prefix match.
    //
    // For a pool whose KV at position i is a function of tokens [0, i] this is 0: a match of
    // m tokens proves all m are reusable. Set it to D when the pool also holds state that
    // reads D tokens ahead -- one-model speculative decoding draft layers -- where a match
    // of m only describes the first m - D positions.
    //
    // Applied inside the match so a single tree walk yields the usable depth.
    int reuseMatchBackoff = 0;

    // Constraint-based memory partitioning.
    std::vector<BatchDesc> constraints;   // batches that must always be supportable
    std::optional<BatchDesc> typicalStep; // typical step for initial ratio computation
    // One normalized hot-tier byte-quota weight per layer group. Cold initialization preserves the implied
    // layer-group slot-count proportions while accounting for cold page sizes.
    std::optional<std::vector<float>> initialPoolRatio; // overrides inferred sizing inputs

    // When set, SWA layers reuse physical pages for out-of-window blocks during prefill.
    // Scratch blocks share coalesced slot sub-pages across blocks for the currently executing
    // layer, reducing peak memory. Trade-off: KV cache reuse is degraded because scratch blocks
    // have no preserved data after the step.
    std::optional<SwaScratchReuseConfig> swaScratchReuse;

    // If true, commit() records only the minimum cache snapshot reusable at the post-call
    // numCommittedTokens. Only the minimum amount of pages required for such reuse will be
    // preserved. Required when SSM layers are present.
    bool commitMinSnapshot = false;

    // Collect V2 KV cache allocation, reuse, and transfer statistics.
    bool enableStats = true;

    // Deployment-level guarantee that no request carries multi-modal content, so token
    // sequences never contain digests. Lets block-key hashing take the digest-free fast
    // path without scanning. A per-KvCache text_only override may only tighten this
    // (a text-only deployment forbids a request claiming otherwise). Default false.
    bool textOnly = false;

    // Publish a finalized partial block for reuse when committing stops.
    // Beam search disables this while retaining full-block reuse.
    bool enablePartialCommit = true;

    bool enableSwaScratchReuse() const noexcept
    {
        return swaScratchReuse.has_value();
    }

    void validate() const;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
