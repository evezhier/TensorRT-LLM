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
// Buffer-relative, carrying no page offset: BufferConfig is an input to the
// KVCacheManager constructor, while a buffer's placement inside a cold page is not
// settled until the cold-page codec is configured during that construction. The manager
// supplies both page offsets, which is why this path requires the default codec, whose
// cold page is a plain concatenation of buffers.
//
// Describes the cold side only. The default codec relocates a buffer's bytes verbatim,
// so a unit is laid out identically in a cold page and in a hot slot; one set of strides
// serves both, with only the unit index differing.
// ---------------------------------------------------------------------------
struct SelectionUnitLayout
{
    uint32_t numChunks = 1;   //!< One per head; 1 for token-major.
    uint32_t chunkBytes = 0;  //!< Contiguous bytes of one unit within one chunk.
    uint32_t chunkStride = 0; //!< Distance between consecutive chunks.
    uint32_t unitStride = 0;  //!< Distance between consecutive selection units.

    //! @param unitSize Bytes of one selection unit, i.e. BufferConfig::size.
    void validate(size_t unitSize) const
    {
        if (numChunks == 0 || chunkBytes == 0)
        {
            throw std::invalid_argument("SelectionUnitLayout: numChunks and chunkBytes must be non-zero");
        }
        if (static_cast<size_t>(numChunks) * chunkBytes != unitSize)
        {
            throw std::invalid_argument("SelectionUnitLayout: numChunks * chunkBytes must equal the buffer size");
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
    size_t size = 0; // bytes per page (without expansion)

    // If set, overrides tokens_per_block for this buffer.
    // Must be a divisor of KVCacheManagerConfig::tokensPerBlock.
    std::optional<int> tokensPerBlockOverride;

    bool isSparse = false; //!< Whether history uses the sparse attention lifecycle.

    //! Selection width: units a sparse-attention step selects per layer. Sparse buffers only.
    int numSelectedBlocks = 0;

    //! Upper bound on per-request GPU cache entries. Two to four times numSelectedBlocks;
    //! a cache only as wide as the selection retains nothing between steps. Sparse buffers only.
    int maxCachedUnits = 0;

    //! Where a selection unit's bytes sit. Leave unset for a buffer whose cold form is
    //! compressed or whose units are not locatable by arithmetic; such a buffer supplies
    //! extraction code instead.
    std::optional<SelectionUnitLayout> sparseUnitLayout;

    void validate() const
    {
        if (!isSparse)
        {
            if (numSelectedBlocks != 0 || maxCachedUnits != 0 || sparseUnitLayout.has_value())
            {
                throw std::invalid_argument("BufferConfig: sparse-only fields set on a non-sparse buffer");
            }
            return;
        }
        // Zero means the fetch path is not configured for this buffer: sparse storage and
        // locking work without it, and the per-request cache rejects it when constructed.
        if (numSelectedBlocks < 0 || maxCachedUnits < 0)
        {
            throw std::invalid_argument("BufferConfig: numSelectedBlocks and maxCachedUnits must not be negative");
        }
        if (numSelectedBlocks > 0 && maxCachedUnits < numSelectedBlocks)
        {
            throw std::invalid_argument("BufferConfig: maxCachedUnits must be at least numSelectedBlocks");
        }
        if (sparseUnitLayout.has_value())
        {
            sparseUnitLayout->validate(size);
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

    [[nodiscard]] std::optional<int> windowSize() const noexcept
    {
        return slidingWindowSize;
    }

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
        for (auto const& buf : buffers)
        {
            if (buf.isSparse != buffers.front().isSparse)
            {
                throw std::invalid_argument("Sparse and non-sparse buffers cannot share an attention layer lifecycle");
            }
            buf.validate();
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
