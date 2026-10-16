# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import replace
from math import ceil, gcd
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

import torch

from tensorrt_llm._torch.pyexecutor import llm_request
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _GUARD_PAGE_REQUEST_ID,
    _RESERVED_REQUEST_IDS,
    CUDA_GRAPH_DUMMY_REQUEST_ID,
    GPU_LEVEL,
    KVCacheManagerV2,
    _fill_kv_pages,
    _get_generation_kv_capacity,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import sparse_cache_config
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import (
    TensorWrapper,
    convert_to_torch_tensor,
    get_size_in_bytes,
    get_sm_version,
    nvtx_range_debug,
    prefer_pinned,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.math_utils import ceil_div
from tensorrt_llm.runtime import ModelConfig
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    BAD_PAGE_INDEX,
    AttentionLayerConfig,
    Batch,
    BatchDesc,
    BufferConfig,
    BufferId,
    CudaStream,
    DataRole,
    KVCacheDesc,
    LayerGroupId,
    LayerId,
    PageIndexMode,
    ScratchDesc,
    SparseCacheConfig,
    _KVCache,
)
from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManagerConfig as KVCacheManagerConfigPy

from .compressor import KVCacheDtype, get_nvfp4_compress_residual_dim
from .kernels import select_sparse_history_units
from .offload import SparseOffloadLayerDescriptor, SparseOffloadPaddingRegion, SparseOffloadState
from .params import (
    DEEPSEEK_V4_NON_SLIDING_ATTENTION,
    DEEPSEEK_V4_SLIDING_ATTENTION,
    DEEPSEEK_V4_SPARSE_RATIO,
    DeepseekV4AttentionType,
    compress_ratio_has_attention,
    is_overlap_compressor,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.interface import AttentionMetadata

COMPRESS_BLOCK_SCALE_ROLE = DataRole("deepseek_v4_compress_block_scale")
KV_CACHE_COPY_ALIGNMENT = 16
# Attention metadata holding a SparseOffloadState: the eager one, and the CUDA
# graph metadata, which share theirs through the graph buffer pool.
SPARSE_OFFLOAD_STATE_COPIES = 2
NVFP4_VECTOR_SIZE = 16


def get_attn_dim(
    head_dim: int, index_head_dim: int, compress_ratio: int, attn_type: DeepseekV4AttentionType
) -> int:
    state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
    if attn_type == DeepseekV4AttentionType.SWA:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESS:
        return head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_KV:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.COMPRESSOR_SCORE:
        return state_factor * head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        return index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV:
        return state_factor * index_head_dim
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE:
        return state_factor * index_head_dim
    raise ValueError(f"Unsupported DeepSeek-V4 attention type: {attn_type}")


def get_token_bytes(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    has_nvfp4_compress: bool = False,
    nvfp4_residual_dim: int | None = None,
) -> int:
    if not compress_ratio_has_attention(compress_ratio, attn_type):
        raise ValueError(
            f"Layer with compress ratio {compress_ratio} does not have attention type {attn_type}"
        )

    if use_fp8_ds_mla and attn_type in (
        DeepseekV4AttentionType.SWA,
        DeepseekV4AttentionType.COMPRESS,
    ):
        from . import footer_scale_kv

        if head_dim != footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE:
            raise ValueError(
                f"footer-scale KV layout requires head_dim "
                f"{footer_scale_kv.DIM_NOPE + footer_scale_kv.DIM_ROPE}, got {head_dim}"
            )
        return footer_scale_kv.TOKEN_BYTES

    attn_dim = get_attn_dim(head_dim, index_head_dim, compress_ratio, attn_type)

    dtype_bytes = 1 if has_fp8_kv_cache else 2
    # (indexer) compressor kv and score always use float32
    if attn_type in [
        DeepseekV4AttentionType.COMPRESSOR_KV,
        DeepseekV4AttentionType.COMPRESSOR_SCORE,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
    ]:
        dtype_bytes = 4  # (indexer) compressor kv and score use float32
    # Indexer cache always packs data + per-block scales into one row.  Only
    # the two indexer presets ("fp8" blockwise / "fp4" mxfp4) are valid
    # here — bf16 / fp8_pertensor are reserved for the main-attention
    # compressor.
    if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
        if indexer_k_dtype == "fp8":
            return attn_dim + index_head_dim // 128 * 4
        if indexer_k_dtype == "fp4":
            return index_head_dim // 2 + index_head_dim // 32
        raise ValueError(
            f"Unsupported indexer_k_dtype {indexer_k_dtype!r}; expected 'fp8' or 'fp4'."
        )

    if attn_type == DeepseekV4AttentionType.COMPRESS and has_nvfp4_compress:
        if nvfp4_residual_dim is None:
            nvfp4_residual_dim = get_nvfp4_compress_residual_dim()
        storage_dim = attn_dim + nvfp4_residual_dim
        return storage_dim // 2 + storage_dim // NVFP4_VECTOR_SIZE

    return attn_dim * dtype_bytes


def _estimate_non_sliding_attn_size_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    has_nvfp4_compress: bool = False,
    nvfp4_residual_dim: int | None = None,
    offload_sparse_history: bool = False,
) -> float:
    """Return the GPU bytes per token of compressed-layer history.

    With ``offload_sparse_history``, size a worker that offloads ratio-4
    history: leave out the offloaded rows and count the rest at their exact
    bytes per token instead of rounding each down to whole bytes.
    """
    total_bytes = 0
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_NON_SLIDING_ATTENTION:
            if (
                offload_sparse_history
                and compress_ratio == DEEPSEEK_V4_SPARSE_RATIO
                and attn_type == DeepseekV4AttentionType.COMPRESS
            ):
                # Offloaded history lives in host memory.
                continue
            if compress_ratio_has_attention(compress_ratio, attn_type):
                total_bytes += _get_attn_bytes_per_token(
                    head_dim,
                    index_head_dim,
                    compress_ratio,
                    attn_type,
                    has_fp8_kv_cache,
                    indexer_k_dtype=indexer_k_dtype,
                    use_fp8_ds_mla=use_fp8_ds_mla,
                    has_nvfp4_compress=has_nvfp4_compress,
                    nvfp4_residual_dim=nvfp4_residual_dim,
                    exact=offload_sparse_history,
                )
    return total_bytes


def _estimate_swa_cache_size(
    head_dim: int,
    index_head_dim: int,
    compress_ratios: List[int],
    has_fp8_kv_cache,
    tokens_per_block: int,
    swa_window_size: int | None,
    *,
    context: bool,
    scratch: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    generation_capacity_headroom: int | None = None,
) -> Tuple[int, int]:
    """Return the sliding-window bytes per token and per request.

    By default a window costs one page per ``tokens_per_block`` tokens of
    window. With ``generation_capacity_headroom``, the tokens a generation step
    reserves above history, it costs the most pages its live interval touches
    at any page offset, which is what the cache manager holds.
    """
    tokens_per_block = int(tokens_per_block)
    size_per_token = 0
    size_per_request = 0
    scratch_keys = set()
    for compress_ratio in compress_ratios:
        for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
            if not compress_ratio_has_attention(compress_ratio, attn_type):
                continue
            if attn_type == DeepseekV4AttentionType.SWA:
                if swa_window_size is None:
                    continue
                window_size = swa_window_size
            else:
                state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
                window_size = state_factor * compress_ratio
            if window_size <= 0:
                continue
            if generation_capacity_headroom is None:
                window_pages = ceil_div(int(window_size), tokens_per_block)
            else:
                # The live interval spans window + headroom - 1 tokens.
                window_pages = (
                    ceil_div(int(window_size) + generation_capacity_headroom - 2, tokens_per_block)
                    + 1
                )
            window_tokens = window_pages * tokens_per_block
            token_bytes = _get_attn_bytes_per_token(
                head_dim,
                index_head_dim,
                compress_ratio,
                attn_type,
                has_fp8_kv_cache,
                indexer_k_dtype=indexer_k_dtype,
                use_fp8_ds_mla=use_fp8_ds_mla,
            )
            if not context:
                size_per_request += window_tokens * token_bytes
            elif not scratch:
                size_per_token += token_bytes
            else:
                scratch_key = (attn_type, compress_ratio)
                if scratch_key in scratch_keys:
                    size_per_request += window_tokens * token_bytes
                else:
                    scratch_keys.add(scratch_key)
                    size_per_token += token_bytes
    return size_per_token, size_per_request


def _get_attn_bytes_per_token(
    head_dim: int,
    index_head_dim: int,
    compress_ratio: int,
    attn_type: DeepseekV4AttentionType,
    has_fp8_kv_cache: bool,
    indexer_k_dtype: str = "fp8",
    use_fp8_ds_mla: bool = False,
    has_nvfp4_compress: bool = False,
    nvfp4_residual_dim: int | None = None,
    exact: bool = False,
) -> float:
    """Return the bytes one token adds to a buffer.

    A compressed row covers ``compress_ratio`` tokens, so its bytes are divided
    among them: rounded down to whole bytes, or exactly when ``exact`` is set.
    """
    token_bytes = get_token_bytes(
        head_dim,
        index_head_dim,
        compress_ratio,
        attn_type,
        has_fp8_kv_cache,
        indexer_k_dtype=indexer_k_dtype,
        use_fp8_ds_mla=use_fp8_ds_mla,
        has_nvfp4_compress=has_nvfp4_compress,
        nvfp4_residual_dim=nvfp4_residual_dim,
    )
    if attn_type in [DeepseekV4AttentionType.COMPRESS, DeepseekV4AttentionType.INDEXER_COMPRESS]:
        if exact:
            return token_bytes / compress_ratio
        token_bytes //= compress_ratio
    return token_bytes


def _get_index_mode(attn_type: DeepseekV4AttentionType) -> PageIndexMode:
    if attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
        return PageIndexMode.PER_LAYER
    else:
        return PageIndexMode.SHARED


class DeepseekV4CacheManager(KVCacheManagerV2):
    _supports_reuse_match_backoff = False

    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_pointers contains one virtual attention-op pool per local
    # SWA layer, shape: [num_local_layers, 2]. The second column is always 0.
    kv_cache_pool_pointers: torch.Tensor
    # This tensor is for compatibility with AttentionOp, it only contains swa attention.
    # kv_cache_pool_mapping contains pool id and layer offset for each layer's swa attention,
    # shape: [num_local_layers, 2]
    kv_cache_pool_mapping: torch.Tensor
    # The block size of the (indexer) compressed cache.
    # For other attention types, block size is tokens_per_block.
    compressed_block_sizes: List[int]
    # Sparse NVFP4 layers whose native write indices were seen to give data and
    # block scales the same arena rows.
    _paired_write_layers: frozenset[int] = frozenset()

    def _get_typical_seq_len(self, kv_cache_config: KvCacheConfig) -> int:
        """Retain DeepSeek-V4's max-length pool-sizing model by default."""
        return (
            kv_cache_config.avg_seq_len
            if kv_cache_config.avg_seq_len is not None
            else self.max_seq_len
        )

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: int = 1,
        max_batch_size: int,
        max_beam_width: int = 1,
        tokens_per_block: int,
        max_seq_len: int,
        vocab_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.BF16,
        compressor_dtype: DataType = DataType.FLOAT,
        sparse_attn_config: Optional[DeepSeekV4SparseAttentionConfig] = None,
        max_input_len: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        if sparse_attn_config is None:
            sparse_attn_config = kwargs.pop("sparse_attention_config", None)
        if sparse_attn_config is None and kwargs.get("model_config") is not None:
            sparse_attn_config = kwargs["model_config"].sparse_attention_config
        if sparse_attn_config is None:
            raise ValueError(
                "sparse_attn_config or sparse_attention_config is required "
                "for DeepseekV4CacheManager"
            )

        self._enable_kv_cache_offload = sparse_attn_config.enable_kv_cache_offload
        self._sparse_offload_topk = sparse_attn_config.index_topk
        self._sparse_batch: Batch | None = None
        self._sparse_batch_members: dict[int, _KVCache] = {}
        self._sparse_batch_prepared = False
        sparse_kv_offload_config = sparse_attn_config.get_kv_cache_offload_config()
        if sparse_kv_offload_config is not None:
            # Report configuration errors before refusing to run.
            self._check_sparse_kv_offload_config(
                sparse_kv_offload_config,
                kv_cache_config,
                mapping,
                spec_config=kwargs.get("spec_config"),
            )
            # Inference requires compressor/attention arena addressing and a
            # prefill-to-decode handoff in addition to the decode fetch adapter.
            raise NotImplementedError(
                "DeepSeek-V4 KV cache offload requires the KVCM v2 sparse runtime "
                "and validated arena read/write addressing, prefill handoff, and transfer ordering."
            )

        # DeepSeek-V4 specific attributes initialization
        assert kv_cache_type == CacheTypeCpp.SELFKONLY, "DeepSeek-V4 only supports SELFKONLY"
        assert num_kv_heads == 1, "DeepSeek-V4 only supports num_kv_heads == 1"
        assert len(sparse_attn_config.compress_ratios) >= num_layers, (
            "The length of compress ratios must be >= the number of layers"
        )
        assert dtype in [DataType.BF16, DataType.FP8, DataType.NVFP4], (
            f"Unsupported dtype: {dtype}, only support BF16, FP8 and NVFP4"
        )
        assert compressor_dtype == DataType.FLOAT, (
            f"Unsupported compressor dtype: {compressor_dtype}, only support FP32/TF32"
        )

        assert tokens_per_block in [128, 256], (
            f"DeepseekV4CacheManager requires tokens_per_block in [128, 256], got {tokens_per_block}. "
            f"Set kv_cache_config.tokens_per_block to 128 or 256."
        )

        self.use_fp8_ds_mla = kv_cache_config.dtype == "fp8_ds_mla"
        sm_version = get_sm_version()
        if sm_version == 90 and not self.use_fp8_ds_mla:
            raise ValueError(
                "DeepSeek-V4 on Hopper requires kv_cache_config.dtype='fp8_ds_mla'; "
                f"got {kv_cache_config.dtype!r}."
            )
        if self.use_fp8_ds_mla and sm_version != 90 and tokens_per_block != 256:
            raise ValueError(
                "DeepSeek-V4 fp8_ds_mla KV cache requires tokens_per_block=256 "
                f"on SM{sm_version}, got {tokens_per_block}."
            )

        self.index_head_dim = sparse_attn_config.index_head_dim
        self._compress_ratios = sparse_attn_config.compress_ratios
        # When MTP is enabled, enlarge the sliding window sizes by
        # max_draft_len so that rewinding rejected draft tokens can still
        # reach the KV entries that would otherwise have been evicted by the
        # sliding-window policy.
        spec_config = kwargs.get("spec_config", None)
        self._max_draft_len = spec_config.max_draft_len if spec_config is not None else 0
        self._swa_window_size = sparse_attn_config.window_size
        self._compressor_dtype = compressor_dtype
        self._use_nvfp4_compress = dtype == DataType.NVFP4
        self._nvfp4_residual_dim = get_nvfp4_compress_residual_dim()
        cache_dtype = DataType.FP8 if self._use_nvfp4_compress else dtype
        # If MTP is enabled, append compress ratios for MTP virtual layers.
        # MTP adds (max_draft_len - 1) extra layers that mirror the last real
        # layer's attention pattern.  Only NEW entries are appended; existing
        # per-layer ratios are never modified, so a real layer with ratio==1
        # stays SWA-only.
        if self._max_draft_len > 0:
            self._compress_ratios = self._compress_ratios + [self._compress_ratios[-1]] * (
                self._max_draft_len - 1
            )
        self.compressed_block_sizes = [tokens_per_block // ratio for ratio in self._compress_ratios]

        self._init_indexer_dtype(sparse_attn_config)

        self._max_input_len = max_input_len
        self._max_num_tokens = max_num_tokens

        # General initialization
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            mapping=mapping,
            dtype=cache_dtype,
            max_num_tokens=max_num_tokens,
            sparse_kv_offload_config=sparse_kv_offload_config,
            **kwargs,
        )
        self.is_vswa = True  # DeepSeek-V4 must has VSWA

        # DeepSeek-V4 expects cache of all layers with the same attention type and compress ratio
        # to be in the same pool and have the same scale.
        self._assert_layer_pool_scale()
        self._assert_nvfp4_compress_pool_layout()
        self._sparse_arena_buffers: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}
        if self._enable_kv_cache_offload:
            descriptors = self.get_sparse_offload_descriptors()
            self._initialize_sparse_arena_buffers(descriptors)
            self._initialize_sparse_arena_padding(descriptors)

        # For DeepSeek-V4 Attention, the base pointer for SWA pool
        # Use first PP layer instead of hardcoded 0 for pipeline parallelism.
        first_pp_layer = self.pp_layers[0]
        self.swa_pool_ptr = self.impl.get_mem_pool_base_address(
            self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA],
            DeepseekV4AttentionType.SWA.role,
            PageIndexMode.PER_LAYER,
        )

        self.compress_pool_ptrs = {}
        self.compress_scale_pool_ptrs = {}
        # Find first PP layer with each compress ratio for pool pointer lookup.
        pp_compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        if 4 in pp_compress_ratios:  # indexer compressor
            first_layer_with_4 = self.pp_layers[pp_compress_ratios.index(4)]
            self.compress_pool_ptrs[4] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[first_layer_with_4, DeepseekV4AttentionType.COMPRESS],
                DeepseekV4AttentionType.COMPRESS.role,
                PageIndexMode.SHARED,
            )
            if self._use_nvfp4_compress:
                self.compress_scale_pool_ptrs[4] = self.impl.get_mem_pool_base_address(
                    self._layer_attn_to_layer_id[
                        first_layer_with_4, DeepseekV4AttentionType.COMPRESS
                    ],
                    COMPRESS_BLOCK_SCALE_ROLE,
                    PageIndexMode.SHARED,
                )
        if 128 in pp_compress_ratios:  # compressor
            first_layer_with_128 = self.pp_layers[pp_compress_ratios.index(128)]
            self.compress_pool_ptrs[128] = self.impl.get_mem_pool_base_address(
                self._layer_attn_to_layer_id[
                    first_layer_with_128, DeepseekV4AttentionType.COMPRESS
                ],
                DeepseekV4AttentionType.COMPRESS.role,
                PageIndexMode.SHARED,
            )
            if self._use_nvfp4_compress:
                self.compress_scale_pool_ptrs[128] = self.impl.get_mem_pool_base_address(
                    self._layer_attn_to_layer_id[
                        first_layer_with_128, DeepseekV4AttentionType.COMPRESS
                    ],
                    COMPRESS_BLOCK_SCALE_ROLE,
                    PageIndexMode.SHARED,
                )

    @property
    def nvfp4_residual_dim(self) -> int:
        """Residual dimensions fixed when the NVFP4 cache is constructed."""
        return self._nvfp4_residual_dim if self._use_nvfp4_compress else 0

    def _assert_nvfp4_compress_pool_layout(self) -> None:
        if not self._use_nvfp4_compress:
            return
        for layer_idx in self.pp_layers:
            compress_ratio = self._compress_ratios[layer_idx]
            if self._enable_kv_cache_offload and compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
                # Arena geometry is validated when building the sparse descriptors;
                # the host page converters do not address the arena.
                continue
            if not compress_ratio_has_attention(compress_ratio, DeepseekV4AttentionType.COMPRESS):
                continue
            layer_id = self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.COMPRESS]
            data_converter = self.impl.get_page_index_converter(
                layer_id, DeepseekV4AttentionType.COMPRESS.role
            )
            scale_converter = self.impl.get_page_index_converter(
                layer_id, COMPRESS_BLOCK_SCALE_ROLE
            )
            data_layout = (
                data_converter.scale,
                data_converter.expansion,
                data_converter.layer_offset,
            )
            scale_layout = (
                scale_converter.scale,
                scale_converter.expansion,
                scale_converter.layer_offset,
            )
            if data_layout != scale_layout:
                raise RuntimeError(
                    "NVFP4 COMPRESS data and scale buffers must use identical "
                    f"page-index conversion; got {data_layout} and {scale_layout}."
                )
            data_pages = self.impl.get_page_index_upper_bound(
                layer_id, DeepseekV4AttentionType.COMPRESS.role
            )
            scale_pages = self.impl.get_page_index_upper_bound(layer_id, COMPRESS_BLOCK_SCALE_ROLE)
            if data_pages != scale_pages:
                raise RuntimeError(
                    "NVFP4 COMPRESS data and scale buffers must have identical "
                    f"page capacity; got {data_pages} and {scale_pages}."
                )

    def _format_kv_cache_pool_lifecycle_entry(self, layer_id: LayerId, role: DataRole) -> str:
        layer_semantics = self._manager_layer_id_to_layer_attn.get((layer_id, role))
        if layer_semantics is None:
            return super()._format_kv_cache_pool_lifecycle_entry(layer_id, role)

        model_layer_idx, attn_type = layer_semantics
        return (
            f"deepseek_role={attn_type.name}, "
            f"compress_ratio={self._compress_ratios[model_layer_idx]}, "
            f"{super()._format_kv_cache_pool_lifecycle_entry(layer_id, role)}"
        )

    def get_buffers(self, layer_idx: int, attn_type: DeepseekV4AttentionType) -> torch.Tensor:
        """
        Get the buffers for a specific layer and attention type.

        Args:
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The buffer tensor (shape: [num_blocks, tokens_per_block, attn_dim])
            For blockwise FP8 layers, shape is [num_blocks, tokens_per_block, attn_dim + scale_size]
            Sparse COMPRESS storage is device-mapped host memory and keeps the ordinary
            page shape, [num_blocks, compressed_block_size, dim_per_token]: a sparse
            buffer's size is its whole page, which the cache divides into selection units.
            GPU arena views are returned by get_sparse_compress_buffers().
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        page_index_mode = _get_index_mode(attn_type)
        addr = self.impl.get_mem_pool_base_address(layer_id, data_role, page_index_mode)

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        attn_dim = get_attn_dim(
            self.head_dim, self.index_head_dim, self._compress_ratios[layer_idx], attn_type
        )
        footer_scale = self.use_fp8_ds_mla and attn_type in (
            DeepseekV4AttentionType.SWA,
            DeepseekV4AttentionType.COMPRESS,
        )
        if attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            # Indexer always pack data + per-block scales into the same row.
            dim_per_token = self._indexer_data_size + self._indexer_scale_size
        elif footer_scale:
            # Footer-scale pages are byte-packed rather than token-major.
            from . import footer_scale_kv

            dim_per_token = footer_scale_kv.TOKEN_BYTES
        elif attn_type == DeepseekV4AttentionType.COMPRESS and self._use_nvfp4_compress:
            dim_per_token = (attn_dim + self.nvfp4_residual_dim) // 2
        else:
            dim_per_token = attn_dim

        page_index_upper_bound = self.impl.get_page_index_upper_bound(layer_id, data_role)
        if page_index_mode == PageIndexMode.PER_LAYER:
            converter = self.impl.get_page_index_converter(layer_id, data_role)
            if converter.layer_offset is not None:
                page_index_upper_bound += converter.layer_offset * converter.expansion

        shape = (page_index_upper_bound, block_size, dim_per_token)

        dtype = self.dtype
        # (indexer) compressor kv and score use compressor_dtype
        if attn_type in [
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ]:
            dtype = self._compressor_dtype
        elif attn_type == DeepseekV4AttentionType.INDEXER_COMPRESS:
            dtype = self._indexer_dtype
        elif footer_scale or (
            attn_type == DeepseekV4AttentionType.COMPRESS and self._use_nvfp4_compress
        ):
            dtype = DataType.UINT8

        return convert_to_torch_tensor(TensorWrapper(addr, dtype, shape))

    def get_compress_scale_buffers(self, layer_idx: int) -> torch.Tensor:
        """Return E4M3 per-16 scales from the NVFP4 COMPRESS storage pool.

        Sparse storage is device-mapped host memory, indexed one compressed row
        at a time. GPU arena scales are returned by get_sparse_compress_buffers().
        """
        if not self._use_nvfp4_compress:
            raise RuntimeError("COMPRESS block scales exist only for NVFP4 KV cache.")
        layer_id = self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.COMPRESS]
        addr = self.impl.get_mem_pool_base_address(
            layer_id, COMPRESS_BLOCK_SCALE_ROLE, PageIndexMode.SHARED
        )
        page_index_upper_bound = self.impl.get_page_index_upper_bound(
            layer_id, COMPRESS_BLOCK_SCALE_ROLE
        )
        block_size = self.compressed_block_sizes[layer_idx]
        shape = (
            page_index_upper_bound,
            block_size,
            (self.head_dim + self.nvfp4_residual_dim) // NVFP4_VECTOR_SIZE,
        )
        return convert_to_torch_tensor(TensorWrapper(addr, DataType.FP8, shape))

    def get_compress_pool_buffers(self, compress_ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return flat NVFP4 data and scale views rooted at their pool bases."""
        if not self._use_nvfp4_compress:
            raise RuntimeError("COMPRESS NVFP4 pools are not enabled.")
        layer_idx = next(
            layer for layer in self.pp_layers if self._compress_ratios[layer] == compress_ratio
        )
        layer_id = self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.COMPRESS]
        data_pages = self.impl.get_page_index_upper_bound(
            layer_id, DeepseekV4AttentionType.COMPRESS.role
        )
        scale_pages = self.impl.get_page_index_upper_bound(layer_id, COMPRESS_BLOCK_SCALE_ROLE)
        tokens_per_page = self.compressed_block_sizes[layer_idx]
        data = convert_to_torch_tensor(
            TensorWrapper(
                self.compress_pool_ptrs[compress_ratio],
                DataType.UINT8,
                [data_pages * tokens_per_page * ((self.head_dim + self.nvfp4_residual_dim) // 2)],
            )
        )
        scales = convert_to_torch_tensor(
            TensorWrapper(
                self.compress_scale_pool_ptrs[compress_ratio],
                DataType.FP8,
                [
                    scale_pages
                    * tokens_per_page
                    * ((self.head_dim + self.nvfp4_residual_dim) // NVFP4_VECTOR_SIZE)
                ],
            )
        )
        return data, scales

    def _get_window_size(
        self, compress_ratio: int, attn_type: DeepseekV4AttentionType
    ) -> int | None:
        if attn_type == DeepseekV4AttentionType.SWA:
            base_window_size = self._swa_window_size
        elif attn_type in (
            DeepseekV4AttentionType.COMPRESSOR_KV,
            DeepseekV4AttentionType.COMPRESSOR_SCORE,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
            DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
        ):
            state_factor = 2 if is_overlap_compressor(compress_ratio) else 1
            base_window_size = state_factor * compress_ratio
        else:
            return None
        return base_window_size + self._max_draft_len

    def _prepare_page_table_tensor(self, index_mapper_capacity: int) -> None:
        # Tensors for compatibility with AttentionOp, only contains swa attention.
        # SWA uses per-layer page indices, so each SWA layer has a virtual
        # attention-op pool.
        # shape: [num_local_layers, 2]
        self.num_attention_op_pools = self.num_local_layers
        self.kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self._layer_attn_to_layer_id[pp_layer, DeepseekV4AttentionType.SWA],
                        DeepseekV4AttentionType.SWA.role,
                        PageIndexMode.PER_LAYER,
                    ),
                    0,
                ]
                for pp_layer in self.pp_layers
            ],
            dtype=torch.int64,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        # shape: [num_local_layers, 2]
        self.kv_cache_pool_mapping = torch.tensor(
            [[local_layer_idx, 0] for local_layer_idx in range(self.num_local_layers)],
            dtype=torch.int32,
            device="cpu",
            pin_memory=prefer_pinned(),
        )
        self.host_kv_cache_block_offsets = torch.empty(
            self.num_pools,
            index_mapper_capacity * self.max_beam_width,
            2,  # key and value
            self.max_blocks_per_seq,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        staging_capacity = self.max_batch_size * self.max_beam_width
        self._host_compress_block_tables_staging = {
            compress_ratio: torch.full(
                (staging_capacity, self.max_blocks_per_seq),
                BAD_PAGE_INDEX,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            for compress_ratio in set(self._compress_ratios)
            if compress_ratio_has_attention(compress_ratio, DeepseekV4AttentionType.COMPRESS)
        }

        # layer offsets per layer and attn, shape [num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)].
        self._layer_offsets = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Pool ids per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_pool_ids = torch.full(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            -1,
            dtype=torch.int32,
            device="cpu",
        )
        # Scales per layer and sliding attention type, shape [num_local_layers, num_sliding_attention_types].
        self._layer_attn_scales = torch.ones(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        # Scratch pages per block per layer and sliding attention type, shape
        # [num_local_layers, num_sliding_attention_types].
        self._scratch_pages = torch.zeros(
            (self.num_local_layers, len(DEEPSEEK_V4_SLIDING_ATTENTION)),
            dtype=torch.int32,
            device="cpu",
        )
        self._csa_compress_pool_id = None
        self._csa_compress_scale = None
        self._csa_indexer_compress_pool_id = None
        self._csa_indexer_compress_scale = None
        self._hca_compress_pool_id = None
        self._hca_compress_scale = None

        for layer_idx in self.pp_layers:
            compress_ratio = self._compress_ratios[layer_idx]
            if compress_ratio == 4:
                compress_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.COMPRESS
                ]
                compress_converter = self.impl.get_page_index_converter(
                    compress_layer_id, DeepseekV4AttentionType.COMPRESS.role
                )
                self._csa_compress_pool_id = self.layer_to_pool_mapping_dict[compress_layer_id]
                self._csa_compress_scale = int(compress_converter.scale)
                indexer_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS
                ]
                indexer_converter = self.impl.get_page_index_converter(
                    indexer_layer_id, DeepseekV4AttentionType.INDEXER_COMPRESS.role
                )
                self._csa_indexer_compress_pool_id = self.layer_to_pool_mapping_dict[
                    indexer_layer_id
                ]
                self._csa_indexer_compress_scale = int(indexer_converter.scale)
            elif compress_ratio == 128:
                compress_layer_id = self._layer_attn_to_layer_id[
                    layer_idx, DeepseekV4AttentionType.COMPRESS
                ]
                compress_converter = self.impl.get_page_index_converter(
                    compress_layer_id, DeepseekV4AttentionType.COMPRESS.role
                )
                self._hca_compress_pool_id = self.layer_to_pool_mapping_dict[compress_layer_id]
                self._hca_compress_scale = int(compress_converter.scale)

            local_layer_idx = self.layer_offsets[layer_idx]
            for attn_type in DEEPSEEK_V4_SLIDING_ATTENTION:
                if not compress_ratio_has_attention(self._compress_ratios[layer_idx], attn_type):
                    continue
                layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
                pool_id = self.layer_to_pool_mapping_dict[layer_id]
                converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
                self._layer_attn_pool_ids[local_layer_idx, attn_type.value] = pool_id
                self._layer_attn_scales[local_layer_idx, attn_type.value] = converter.scale
                self._layer_offsets[local_layer_idx, attn_type.value] = converter.layer_offset
                self._scratch_pages[local_layer_idx, attn_type.value] = (
                    converter.scratch_pages_per_block
                )

        device = torch.device("cuda", torch.cuda.current_device())
        self._device_kv_cache_block_offsets_input = torch.empty_like(
            self.host_kv_cache_block_offsets,
            device=device,
        )
        self._precomputed_sliding_block_tables = torch.empty(
            (
                self.num_local_layers,
                len(DEEPSEEK_V4_SLIDING_ATTENTION),
                self.host_kv_cache_block_offsets.size(1),
                self.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device=device,
        )
        self._device_copy_idx_staging = torch.zeros(
            self.host_kv_cache_block_offsets.size(1),
            dtype=torch.int32,
            device=device,
        )
        self._device_num_contexts = torch.empty((), dtype=torch.int32, device=device)
        self._device_layer_offsets = self._layer_offsets.to(device=device)
        self._device_layer_attn_pool_ids = self._layer_attn_pool_ids.to(
            device=device,
            dtype=torch.long,
        )
        self._device_layer_attn_scales = self._layer_attn_scales.to(device=device)
        self._device_scratch_pages = self._scratch_pages.to(device=device)
        self._device_valid_sliding_pool = self._device_layer_attn_pool_ids >= 0
        self._device_block_positions = torch.arange(
            self.max_blocks_per_seq,
            dtype=torch.int32,
            device=device,
        )

        if self.enable_swa_scratch_reuse:
            valid_scales = self._layer_attn_scales[self._layer_attn_pool_ids >= 0]
            min_scale = int(valid_scales.min().item()) if valid_scales.numel() > 0 else 1
            max_scratch_pages = int(self._scratch_pages.max().item())
            self._max_scratch_slots = max(
                1,
                (self.max_blocks_per_seq * max_scratch_pages + min_scale - 1) // min_scale,
            )
            scratch_slots_shape = (
                self.num_pools,
                staging_capacity,
                self._max_scratch_slots,
            )
            self._host_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._host_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                pin_memory=prefer_pinned(),
                device="cpu",
            )
            self._device_scratch_begs_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_ends_staging = torch.empty(
                self.num_pools,
                staging_capacity,
                dtype=torch.int32,
                device=device,
            )
            self._device_scratch_slots_staging = torch.empty(
                scratch_slots_shape,
                dtype=torch.int32,
                device=device,
            )

    @property
    def blocks_in_primary_pool(self) -> int:
        first_pp_layer = self.pp_layers[0]
        swa_layer_id = self._layer_attn_to_layer_id[first_pp_layer, DeepseekV4AttentionType.SWA]
        return self.impl.get_page_index_upper_bound(swa_layer_id, DeepseekV4AttentionType.SWA.role)

    def get_num_free_blocks(self) -> int:
        # This method reports primary-pool capacity while the manager holds no
        # request. The guard page, when enabled, is held by a permanent
        # reservation made at construction (not by a request), so exclude it
        # from the emptiness check and subtract the pages it owns -- a caller
        # sizing off this number must not plan to use the guard's page (mirrors
        # the base get_num_free_blocks). DSV4 allocates no generic Role.KEY
        # buffer, so use SWA's model-specific DataRole for the estimate.
        reserved = set(self.kv_cache_map) & _RESERVED_REQUEST_IDS
        assert not set(self.kv_cache_map) - reserved, (
            "get_num_free_blocks is only used when the kv cache manager is empty"
        )
        max_num_pages = max(
            self.impl.get_page_index_upper_bound(
                self._layer_attn_to_layer_id[layer_idx, DeepseekV4AttentionType.SWA],
                DeepseekV4AttentionType.SWA.role,
            )
            for layer_idx in self.pp_layers
        )
        reserved_pages = sum(int(self.kv_cache_map[req_id].num_blocks) for req_id in reserved)
        return max_num_pages - reserved_pages

    def get_cache_indices(
        self,
        request_id: int,
        layer_idx: int,
        attn_type: DeepseekV4AttentionType,
    ) -> List[int]:
        """
        Get the cache block indices for a batch of requests at a specific layer and attention type.

        Args:
            request_id: The request id
            layer_idx: The layer index
            attn_type: The attention type

        Returns:
            The cache block indices, shape (max_blocks_per_seq,)
        """
        layer_id = self._layer_attn_to_layer_id[(layer_idx, attn_type)]
        data_role = attn_type.role
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        kv_cache = self.kv_cache_map[request_id]
        base_indices = kv_cache.get_base_page_indices(pool_id).tolist()
        converter = self.impl.get_page_index_converter(layer_id, data_role)
        page_index_mode = _get_index_mode(attn_type)
        return converter(
            base_indices,
            page_index_mode,
            kv_cache.get_scratch_desc(pool_id),
        )

    def _get_window_page_headroom(self) -> int | None:
        """Headroom for counting the pages a window holds with offload, else None."""
        if not self._enable_kv_cache_offload:
            return None
        return self._generation_kv_capacity_headroom

    def _get_extra_quota_padding(self) -> int:
        """Ensure each attention type has minimal space when max_tokens is small."""
        return len(DeepseekV4AttentionType) * (2 << 20)

    def _get_quota_from_max_tokens(self, max_tokens: int) -> int:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            has_nvfp4_compress=self._use_nvfp4_compress,
            nvfp4_residual_dim=self.nvfp4_residual_dim,
            offload_sparse_history=self._enable_kv_cache_offload,
        )
        (
            context_swa_size_per_token,
            _,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            generation_capacity_headroom=self._get_window_page_headroom(),
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            generation_capacity_headroom=self._get_window_page_headroom(),
        )
        max_context_tokens = (
            self._max_num_tokens if self._max_num_tokens is not None else max_tokens
        )
        context_tokens = min(max_tokens, max_context_tokens)
        generation_tokens = max_tokens - context_tokens
        generation_quota = (
            max_tokens * non_sliding_attn_size_per_token
            + generation_tokens * generation_swa_size_per_token
            + self.max_batch_size * generation_swa_size_per_request
        )
        context_extra_quota = context_tokens * context_swa_size_per_token
        padding = self._get_extra_quota_padding()
        return int(generation_quota + context_extra_quota + padding)

    def _get_max_tokens_from_quota(self, quota: int) -> float:
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        has_fp8_kv_cache = self.dtype == DataType.FP8
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            has_nvfp4_compress=self._use_nvfp4_compress,
            nvfp4_residual_dim=self.nvfp4_residual_dim,
            offload_sparse_history=self._enable_kv_cache_offload,
        )
        context_swa_size_per_token, _ = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=True,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            generation_capacity_headroom=self._get_window_page_headroom(),
        )
        (
            generation_swa_size_per_token,
            generation_swa_size_per_request,
        ) = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            generation_capacity_headroom=self._get_window_page_headroom(),
        )
        padding = self._get_extra_quota_padding()
        size_per_batch = self.max_batch_size * generation_swa_size_per_request + padding
        if quota < size_per_batch:
            return 0
        context_size_per_token = non_sliding_attn_size_per_token + context_swa_size_per_token
        if self._max_num_tokens is None:
            return (quota - size_per_batch) / context_size_per_token

        context_limit_quota = self._max_num_tokens * context_size_per_token + size_per_batch
        if quota <= context_limit_quota:
            return (quota - size_per_batch) / context_size_per_token

        generation_size_per_token = non_sliding_attn_size_per_token + generation_swa_size_per_token
        if generation_size_per_token <= 0:
            return float("inf")
        return self._max_num_tokens + (quota - context_limit_quota) / generation_size_per_token

    def _build_cache_config(self, config: KVCacheManagerConfigPy) -> KVCacheManagerConfigPy:
        """
        Add DeepSeek-V4 layers and warmup constraints to the cache config.
        """
        layers: List[AttentionLayerConfig] = []
        layer_attn_to_layer_id: Dict[Tuple[int, DeepseekV4AttentionType], LayerId] = {}
        manager_layer_id_to_layer_attn: Dict[
            Tuple[LayerId, DataRole], Tuple[int, DeepseekV4AttentionType]
        ] = {}
        disagg_ignored_roles: set[DataRole] = set()
        nvfp4_padding_pages: Dict[int, int] = {}
        if self._use_nvfp4_compress:
            compress_layers_by_scale_page_size: Dict[int, List[int]] = defaultdict(list)
            scale_bytes_per_token = (self.head_dim + self.nvfp4_residual_dim) // NVFP4_VECTOR_SIZE
            for layer_idx in self.pp_layers:
                compress_ratio = self._compress_ratios[layer_idx]
                if compress_ratio_has_attention(compress_ratio, DeepseekV4AttentionType.COMPRESS):
                    scale_page_size = self.compressed_block_sizes[layer_idx] * scale_bytes_per_token
                    compress_layers_by_scale_page_size[scale_page_size].append(layer_idx)

            for scale_page_size, grouped_layers in compress_layers_by_scale_page_size.items():
                page_count_alignment = KV_CACHE_COPY_ALIGNMENT // gcd(
                    scale_page_size, KV_CACHE_COPY_ALIGNMENT
                )
                padding_pages = (-len(grouped_layers)) % page_count_alignment
                if padding_pages:
                    nvfp4_padding_pages[grouped_layers[-1]] = padding_pages

        def _make_buffer_config(
            layer_idx: int,
            role: DataRole,
            page_bytes: int,
            *,
            is_sparse: bool,
        ) -> BufferConfig:
            if not is_sparse:
                return BufferConfig(role=role, size=page_bytes)
            if self.use_fp8_ds_mla:
                raise NotImplementedError(
                    "Sparse fp8_ds_mla compressed KV requires a selection layout with "
                    "separate data and footer-scale regions"
                )
            # `size` is the whole page, as for any other buffer; the layer's sparse_cache
            # says how many selection units it divides into. num_kv_heads is 1 here, so a
            # unit is one contiguous range and the default sparse_unit_chunks of 1 is right.
            return BufferConfig(role=role, size=page_bytes, is_sparse=True)

        def _sparse_cache_config() -> SparseCacheConfig:
            topk = self._sparse_offload_topk
            if topk is None or topk <= 0:
                raise ValueError("Sparse compressed KV requires a positive index_topk")
            # A selection names compressed rows of four original tokens each.
            return sparse_cache_config(
                num_selected_units=topk,
                unit_tokens=DEEPSEEK_V4_SPARSE_RATIO,
                tokens_per_block=self.tokens_per_block,
                gpu_cache_factor=self._sparse_kv_offload.gpu_cache_factor,
            )

        def _add_layer(
            layer_idx: int,
            attention_types: List[DeepseekV4AttentionType],
            sliding_window_size: int | None,
            *,
            sparse_roles: frozenset[DeepseekV4AttentionType] = frozenset(),
        ) -> None:
            # Sparsity is per buffer, so one layer may hold the compressed KV the cache
            # fetches through beside the indexer that selects over it -- they are one
            # attention computation and belong together.
            def _is_sparse(attn_type: DeepseekV4AttentionType) -> bool:
                return attn_type in sparse_roles

            layer_id = LayerId(len(layers))
            for attn_type in attention_types:
                layer_attn_to_layer_id[layer_idx, attn_type] = layer_id
                manager_layer_id_to_layer_attn[layer_id, attn_type.role] = (
                    layer_idx,
                    attn_type,
                )
            buffers = []
            for attn_type in attention_types:
                size = self._get_attn_bytes_per_block(attn_type, layer_idx)
                buffers.append(
                    _make_buffer_config(
                        layer_idx, attn_type.role, size, is_sparse=_is_sparse(attn_type)
                    )
                )
            if self._use_nvfp4_compress and DeepseekV4AttentionType.COMPRESS in attention_types:
                scale_page_size = (
                    self.compressed_block_sizes[layer_idx]
                    * (self.head_dim + self.nvfp4_residual_dim)
                    // NVFP4_VECTOR_SIZE
                )
                buffers.append(
                    _make_buffer_config(
                        layer_idx,
                        COMPRESS_BLOCK_SCALE_ROLE,
                        scale_page_size,
                        is_sparse=_is_sparse(DeepseekV4AttentionType.COMPRESS),
                    )
                )
                # Reuse snapshots copy complete coalesced slots in 16-byte
                # grains. Pad with complete data/scale pages so the shared-page
                # stride and the data/scale page-index converters stay equal.
                data_page_size = self._get_attn_bytes_per_block(
                    DeepseekV4AttentionType.COMPRESS, layer_idx
                )
                for padding_idx in range(nvfp4_padding_pages.get(layer_idx, 0)):
                    data_padding_role = DataRole(f"deepseek_v4_compress_padding_{padding_idx}")
                    scale_padding_role = DataRole(
                        f"deepseek_v4_compress_block_scale_padding_{padding_idx}"
                    )
                    buffers.extend(
                        [
                            _make_buffer_config(
                                layer_idx,
                                data_padding_role,
                                data_page_size,
                                is_sparse=_is_sparse(DeepseekV4AttentionType.COMPRESS),
                            ),
                            _make_buffer_config(
                                layer_idx,
                                scale_padding_role,
                                scale_page_size,
                                is_sparse=_is_sparse(DeepseekV4AttentionType.COMPRESS),
                            ),
                        ]
                    )
                    disagg_ignored_roles.update((data_padding_role, scale_padding_role))
            layer_config = AttentionLayerConfig(
                layer_id=layer_id,
                buffers=buffers,
                sliding_window_size=sliding_window_size,
                num_sink_tokens=None,
                sparse_cache=_sparse_cache_config() if sparse_roles else None,
            )
            layers.append(layer_config)

        # create the layer config for DeepSeek-V4
        for layer in self.pp_layers:
            compress_ratio = self._compress_ratios[layer]
            if compress_ratio == 1:
                _add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
            elif compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
                _add_layer(
                    layer,
                    [DeepseekV4AttentionType.SWA],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                # One layer either way. Only the compressed KV is fetched through the
                # sparse cache; the indexer that selects over it stays device-resident.
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESS,
                        DeepseekV4AttentionType.INDEXER_COMPRESS,
                    ],
                    None,
                    sparse_roles=(
                        frozenset({DeepseekV4AttentionType.COMPRESS})
                        if self._enable_kv_cache_offload
                        else frozenset()
                    ),
                )
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_KV,
                        DeepseekV4AttentionType.INDEXER_COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.COMPRESSOR_KV),
                )
            elif compress_ratio == 128:
                _add_layer(
                    layer,
                    [
                        DeepseekV4AttentionType.SWA,
                        DeepseekV4AttentionType.COMPRESSOR_KV,
                        DeepseekV4AttentionType.COMPRESSOR_SCORE,
                    ],
                    self._get_window_size(compress_ratio, DeepseekV4AttentionType.SWA),
                )
                _add_layer(layer, [DeepseekV4AttentionType.COMPRESS], None)
            else:
                raise ValueError(f"Unsupported DeepSeek-V4 compress ratio {compress_ratio}.")

        # the mapping from layer index and attention type to layer id
        self._layer_attn_to_layer_id = layer_attn_to_layer_id
        self._manager_layer_id_to_layer_attn = manager_layer_id_to_layer_attn
        self._disagg_ignored_roles = frozenset(disagg_ignored_roles)
        # number of layers in the KVCacheManagerPy
        self._num_manager_layers = len(layers)

        constraints = list(config.constraints)
        # _build_base_config copies kv_cache_config.pool_ratio to initial_pool_ratio.
        if config.initial_pool_ratio is None:
            # DeepSeek-V4's windowed and compressed pools must also support
            # the longest decode request alongside the short decode requests.
            constraint_batch_size = self._get_generation_request_capacity()
            if (
                self.max_cuda_graph_batch_size is not None
                and self.max_cuda_graph_batch_size > 0
                and self.is_estimating_kv_cache
                and all(window is None for window in self.max_attention_window_vec)
            ):
                constraint_batch_size = min(constraint_batch_size, self.max_cuda_graph_batch_size)
            constraint_batch_size = max(1, constraint_batch_size)
            min_decode_capacity = 1 + self.max_draft_len + self.num_extra_kv_tokens
            constraints.append(
                BatchDesc(
                    [
                        KVCacheDesc(
                            capacity=self.max_seq_len,
                            history_length=self.max_seq_len - 1,
                        )
                    ]
                    + [KVCacheDesc(capacity=min_decode_capacity, history_length=0)]
                    * (constraint_batch_size - 1)
                )
            )
        return replace(
            config,
            layers=layers,
            constraints=constraints,
        )

    def _init_indexer_dtype(self, sparse_attn_config: DeepSeekV4SparseAttentionConfig) -> None:
        # Indexer compressor cache layout. Two modes are supported:
        #   - "fp8" (FP8 blockwise): 1 byte per value + 1 fp32 scale per 128
        #     values.
        #   - "fp4" (MXFP4 blockwise): ½ byte per value (two FP4 codes packed
        #     per byte) + 1 ue8m0 byte per 32 values. At index_head_dim=128
        #     this halves the per-token indexer-K footprint vs FP8.
        self._indexer_k_dtype = sparse_attn_config.indexer_k_dtype
        if self._indexer_k_dtype == "fp8":
            self._indexer_cache_dtype = KVCacheDtype.FP8_BLOCKWISE
            self._indexer_dtype = DataType.FP8
            self.quant_block_size = 128
            self._indexer_data_size = self.index_head_dim
            self._indexer_scale_size = get_size_in_bytes(
                self.index_head_dim // self.quant_block_size, DataType.FLOAT
            )
        elif self._indexer_k_dtype == "fp4":
            assert self.index_head_dim == 128, (
                f"FP4 indexer K cache requires index_head_dim=128, got {self.index_head_dim}."
            )
            self._indexer_cache_dtype = KVCacheDtype.MXFP4_BLOCKWISE
            # Pool dtype is uint8 because PyTorch can't allocate float4
            # backing storage; downstream consumers reinterpret these
            # raw bytes as packed E2M1 + UE8M0 exponents.
            self._indexer_dtype = DataType.UINT8
            self.quant_block_size = 32
            # Two E2M1 codes pack into one byte → half the data footprint.
            self._indexer_data_size = self.index_head_dim // 2
            # 1 UE8M0 byte per 32-element block.
            self._indexer_scale_size = self.index_head_dim // self.quant_block_size
        else:
            raise ValueError(
                f"Unsupported indexer_k_dtype "
                f"{sparse_attn_config.indexer_k_dtype!r}; expected "
                "'fp8' or 'fp4'."
            )
        # FP4 indexer flag mirrors `DSACacheManager.use_fp4` so the shared
        # base Indexer can branch without knowing the V4-specific enum.
        self.use_fp4 = self._indexer_k_dtype == "fp4"
        assert self.index_head_dim % self.quant_block_size == 0, (
            f"indexer_head_dim {self.index_head_dim} must be divisible by {self.quant_block_size}"
        )

    def _assert_layer_pool_scale(self) -> None:
        attn_ratio_to_pool_id = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})
        attn_ratio_to_scale = defaultdict[DeepseekV4AttentionType, dict[int, int]](lambda: {})

        comb = [
            (attn_type, layer_idx)
            for attn_type in DeepseekV4AttentionType
            for layer_idx in self.pp_layers
            if compress_ratio_has_attention(self._compress_ratios[layer_idx], attn_type)
        ]
        for attn_type, layer_idx in comb:
            compress_ratio = self._compress_ratios[layer_idx]
            layer_id = self._layer_attn_to_layer_id[layer_idx, attn_type]
            pool_id = self.layer_to_pool_mapping_dict[layer_id]
            converter = self.impl.get_page_index_converter(layer_id, attn_type.role)
            assert converter.expansion == 1, "DeepSeek-V4 page index expansion must be 1"
            scale = converter.scale

            # check if the pool id is consistent
            if compress_ratio in attn_ratio_to_pool_id[attn_type]:
                other_pool_id = attn_ratio_to_pool_id[attn_type][compress_ratio]
                assert other_pool_id == pool_id, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has pool id {pool_id}, "
                    f"but another layer with the same compress ratio and attention type has pool id {other_pool_id}."
                    "DeepSeek-V4 expects they share the same pool."
                )
            else:
                attn_ratio_to_pool_id[attn_type][compress_ratio] = pool_id

            # check if the scale is consistent
            if compress_ratio in attn_ratio_to_scale[attn_type]:
                other_scale = attn_ratio_to_scale[attn_type][compress_ratio]
                assert other_scale == scale, (
                    f"Layer {layer_idx} with compress ratio {compress_ratio}, "
                    f"its attention type {attn_type.name} has scale {scale}, "
                    f"but another layer with the same compress ratio and attention type has scale {other_scale}."
                    "DeepSeek-V4 expects they share the same scale."
                )
            else:
                attn_ratio_to_scale[attn_type][compress_ratio] = scale

        # check if all swa attentions are in the same pool and have the same scale
        swa_pool_ids = set(attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA].values())
        swa_scales = set(attn_ratio_to_scale[DeepseekV4AttentionType.SWA].values())
        assert len(swa_pool_ids) == 1, "All swa attentions must be in the same pool"
        assert len(swa_scales) == 1, "All swa attentions must have the same scale"

        # Ensure all compress ratios have SWA entries, not just PP-local ones.
        # The attention metadata uses compress_ratio=1 as a hardcoded SWA key,
        # but with pipeline parallelism a PP stage may not have any layers with
        # ratio 1. Since all SWA layers share the same pool, we populate entries
        # for every compress ratio in the model.
        swa_pool_id = next(iter(swa_pool_ids))
        swa_scale = next(iter(swa_scales))
        for ratio in set(self._compress_ratios):
            if ratio not in attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA]:
                attn_ratio_to_pool_id[DeepseekV4AttentionType.SWA][ratio] = swa_pool_id
                attn_ratio_to_scale[DeepseekV4AttentionType.SWA][ratio] = swa_scale

    def _get_attn_bytes_per_block(
        self,
        attn_type: DeepseekV4AttentionType,
        layer_idx: int,
    ) -> int:
        """
        Get the cache bytes per block for a specific attention type and layer.
        """
        has_fp8_kv_cache = self.dtype == DataType.FP8
        token_bytes = get_token_bytes(
            self.head_dim,
            self.index_head_dim,
            self._compress_ratios[layer_idx],
            attn_type,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            has_nvfp4_compress=self._use_nvfp4_compress,
            nvfp4_residual_dim=self.nvfp4_residual_dim,
        )

        if attn_type == DeepseekV4AttentionType.COMPRESS and self._use_nvfp4_compress:
            token_bytes = (self.head_dim + self.nvfp4_residual_dim) // 2

        block_size = self.tokens_per_block
        if attn_type in [
            DeepseekV4AttentionType.COMPRESS,
            DeepseekV4AttentionType.INDEXER_COMPRESS,
        ]:
            block_size = self.compressed_block_sizes[layer_idx]

        return token_bytes * block_size

    def get_cache_bytes_per_token(self) -> int:
        """Get the average cache bytes per token for DeepSeek-V4."""
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        return ceil(
            _estimate_non_sliding_attn_size_per_token(
                self.head_dim,
                self.index_head_dim,
                compress_ratios,
                has_fp8_kv_cache,
                indexer_k_dtype=self._indexer_k_dtype,
                use_fp8_ds_mla=self.use_fp8_ds_mla,
                has_nvfp4_compress=self._use_nvfp4_compress,
                nvfp4_residual_dim=self.nvfp4_residual_dim,
                offload_sparse_history=self._enable_kv_cache_offload,
            )
        )

    def get_max_resource_count(self) -> int:
        # Keep scheduler capacity tied to physical GPU KV quota in bytes.
        return int(self.impl.get_quota(GPU_LEVEL))

    def _is_context_request(self, request: llm_request.LlmRequest) -> bool:
        if request.is_context_init_state:
            return True
        return request.state == llm_request.LlmRequestState.CONTEXT_INIT

    def _is_generation_request(self, request: llm_request.LlmRequest) -> bool:
        if (
            request.is_generation_in_progress_state
            or request.is_generation_to_complete_state
            or request.is_disagg_generation_init_state
        ):
            return True
        return request.state in (
            llm_request.LlmRequestState.GENERATION_IN_PROGRESS,
            llm_request.LlmRequestState.GENERATION_TO_COMPLETE,
        )

    def _get_context_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        total_tokens = prompt_len + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=True)

    def _get_generation_bytes(self, request: llm_request.LlmRequest) -> int:
        prompt_len = max(0, request.prompt_len)
        max_new_tokens = max(0, request.max_new_tokens)
        total_tokens = prompt_len + max_new_tokens + self.num_extra_kv_tokens
        return self._get_cache_bytes_for_tokens(total_tokens, context=False)

    def _get_cache_bytes_for_tokens(self, total_tokens: int, *, context: bool) -> int:
        has_fp8_kv_cache = self.dtype == DataType.FP8
        compress_ratios = [self._compress_ratios[layer] for layer in self.pp_layers]
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            has_nvfp4_compress=self._use_nvfp4_compress,
            nvfp4_residual_dim=self.nvfp4_residual_dim,
            offload_sparse_history=self._enable_kv_cache_offload,
        )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            self.head_dim,
            self.index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            self.tokens_per_block,
            self._swa_window_size,
            context=context,
            scratch=self.enable_swa_scratch_reuse,
            indexer_k_dtype=self._indexer_k_dtype,
            use_fp8_ds_mla=self.use_fp8_ds_mla,
            generation_capacity_headroom=self._get_window_page_headroom(),
        )
        return int(
            total_tokens * (non_sliding_attn_size_per_token + swa_size_per_token)
            + swa_size_per_request
        )

    def get_request_device_bytes(self, num_tokens: int) -> int:
        """Return the device pool bytes one decoding request of ``num_tokens`` tokens holds."""
        return self._get_cache_bytes_for_tokens(num_tokens, context=False)

    def get_sparse_offload_scratch_bytes(self, config: KVCacheManagerConfigPy) -> int:
        """Return the device bytes of the attention metadata's sparse offload state.

        Counts a copy for the CUDA graph metadata even when graphs are off.
        """
        buffer_counts = {
            sum(buffer.is_sparse for buffer in layer.buffers)
            for layer in config.layers
            if getattr(layer, "sparse_cache", None) is not None
        }
        if not buffer_counts:
            return 0
        return SPARSE_OFFLOAD_STATE_COPIES * SparseOffloadState.device_bytes(
            max_batch_size=self.max_batch_size,
            topk=self._sparse_offload_topk,
            buffer_counts=buffer_counts,
            nvfp4_head_dim=self.head_dim if self._use_nvfp4_compress else None,
        )

    def get_needed_resource_to_completion(self, request: llm_request.LlmRequest) -> int:
        if self._is_generation_request(request):
            return self._get_generation_bytes(request)
        if self._is_context_request(request):
            return self._get_context_bytes(request)
        raise ValueError(f"Unsupported request state: {request.state}")

    def get_layer_bytes_per_token(
        self,
        local_layer_idx: int,
        data_role: DataRole,
    ) -> int:
        # The generic layers in the base config are replaced by
        # _build_cache_config, so their buffer sizes are only placeholders.
        return 1

    def get_disagg_ignored_roles(self) -> frozenset[DataRole]:
        """Exclude storage-only NVFP4 alignment pages from KV transfer."""
        return self._disagg_ignored_roles

    def get_indexer_k_cache_buffers(self, layer_idx: int) -> torch.Tensor:
        """
        Get the buffers for the indexer k cache for a specific layer.
        """
        buffer = self.get_buffers(layer_idx, DeepseekV4AttentionType.INDEXER_COMPRESS).unsqueeze(2)
        return buffer.view(torch.uint8)

    def _compute_shared_block_table(
        self, pool_id: int, scale: int, copy_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the shared offset for one pool and copy index.
        Return shape: [num_seqs, max_blocks_per_seq]
        """
        base = self.host_kv_cache_block_offsets[pool_id, copy_idx, 0, :]
        return torch.where(base == BAD_PAGE_INDEX, BAD_PAGE_INDEX, base * scale)

    def _copy_idx_to_device(self, copy_idx: torch.Tensor) -> torch.Tensor:
        num_tables = copy_idx.size(0)
        device_copy_idx = self._device_copy_idx_staging[:num_tables]
        device_copy_idx.copy_(copy_idx, non_blocking=True)
        # Keep the compiled graph independent of the active table count.
        return self._device_copy_idx_staging

    def _copy_scratch_metadata_to_device(
        self,
        scratch_descs_by_pool: list[list[ScratchDesc | None]],
        num_contexts: int,
        host_begs_staging: torch.Tensor,
        host_ends_staging: torch.Tensor,
        host_slots_staging: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # shape: [num_pools, num_contexts]
        host_begs = host_begs_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts]
        host_ends = host_ends_staging[:, :num_contexts]
        # shape: [num_pools, num_contexts, num_slots]
        host_slots = host_slots_staging[:, :num_contexts, :]
        host_begs.zero_()
        host_ends.zero_()
        host_slots.zero_()

        for pool_idx, scratch_descs in enumerate(scratch_descs_by_pool):
            for context_idx, desc in enumerate(scratch_descs):
                if desc is None:
                    continue
                slot_ids = desc.slot_ids
                if len(slot_ids) > self._max_scratch_slots:
                    raise RuntimeError(
                        f"Scratch slot count {len(slot_ids)} exceeds staging capacity "
                        f"{self._max_scratch_slots}"
                    )
                host_begs[pool_idx, context_idx] = int(desc.range.beg)
                host_ends[pool_idx, context_idx] = int(desc.range.end)
                for slot_idx, slot_id in enumerate(slot_ids):
                    host_slots[pool_idx, context_idx, slot_idx] = int(slot_id)

        self._device_scratch_begs_staging.copy_(host_begs_staging, non_blocking=True)
        self._device_scratch_ends_staging.copy_(host_ends_staging, non_blocking=True)
        self._device_scratch_slots_staging.copy_(host_slots_staging, non_blocking=True)
        # Keep scratch tensor shapes fixed; the device-side mask gates num_contexts.
        return (
            self._device_scratch_begs_staging,
            self._device_scratch_ends_staging,
            self._device_scratch_slots_staging,
        )

    @nvtx_range_debug("dsv4_compute_sliding_block_tables")
    def compute_sliding_block_tables(
        self,
        request_ids: List[int],
        num_contexts: int,
    ) -> None:
        """Compute all per-layer sliding-window block tables for this batch."""
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, 1)
        num_tables = copy_idx.size(0)
        self._num_tables = num_tables

        scratch_descs_by_pool = None
        if self.enable_swa_scratch_reuse and num_contexts > 0:
            scratch_descs_by_pool = [
                [
                    self.kv_cache_map[req].get_scratch_desc(pool_id)
                    for req in request_ids[:num_contexts]
                ]
                for pool_id in range(self.num_pools)
            ]

        device_copy_idx = self._copy_idx_to_device(copy_idx)
        self._device_kv_cache_block_offsets_input.copy_(
            self.host_kv_cache_block_offsets,
            non_blocking=True,
        )

        if scratch_descs_by_pool is not None:
            scratch_begs, scratch_ends, scratch_slots = self._copy_scratch_metadata_to_device(
                scratch_descs_by_pool,
                num_contexts,
                self._host_scratch_begs_staging,
                self._host_scratch_ends_staging,
                self._host_scratch_slots_staging,
            )
            self._device_num_contexts.fill_(num_contexts)
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables_with_scratch(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._device_scratch_pages,
                scratch_begs,
                scratch_ends,
                scratch_slots,
                self._device_num_contexts,
                self._precomputed_sliding_block_tables,
            )
        else:
            torch.ops.trtllm.deepseek_v4_compute_sliding_block_tables(
                self._device_kv_cache_block_offsets_input,
                device_copy_idx,
                self._device_layer_attn_pool_ids,
                self._device_valid_sliding_pool,
                self._device_layer_attn_scales,
                self._device_layer_offsets,
                self._precomputed_sliding_block_tables,
            )

    @nvtx_range_debug("dsv4_copy_batch_block_offsets")
    def copy_batch_block_offsets(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
        max_blocks: Optional[int] = None,
    ) -> None:
        """For compatibility with AttentionOp, copy only the SWA block offsets.

        max_blocks is accepted for signature parity with KVCacheManager; the
        copy below is already bounded by the precomputed SWA table width.
        """
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        assert dst_tensor.is_cuda, "copy_batch_block_offsets expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :num_seqs, 0, :].copy_(
            self._precomputed_sliding_block_tables[
                :, DeepseekV4AttentionType.SWA.value, :num_seqs, :
            ],
            non_blocking=True,
        )

    @nvtx_range_debug("dsv4_copy_batch_sliding_block_tables")
    def copy_batch_sliding_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """
        Copy the per-layer block tables for attentions managed in sliding-window mode to the GPU tensor.
        """
        assert dst_tensor.is_cuda, "copy_batch_sliding_block_tables expects a CUDA destination"
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :, :num_seqs, :].copy_(
            self._precomputed_sliding_block_tables[:, :, :num_seqs, :],
            non_blocking=True,
        )

    def prepare_resources(self, scheduled_batch: ScheduledRequests) -> None:
        self._release_sparse_batch()
        if self._enable_kv_cache_offload and any(
            not (request.is_first_context_chunk and request.is_last_context_chunk)
            for request in scheduled_batch.context_requests
        ):
            # Attention metadata's prompt_lens contains the current chunk's
            # length, so only the scheduled request can identify a first chunk
            # that does not complete its prompt. Reject before model writes.
            raise NotImplementedError("Sparse offload does not support chunked prefill")
        super().prepare_resources(scheduled_batch)
        if not self._enable_kv_cache_offload:
            return
        requests = scheduled_batch.context_requests + scheduled_batch.generation_requests
        # Disaggregated gen-init requests are prepared before their KV transfer
        # starts and run no forward this step, so they never join the Batch. A
        # batch holding only those publishes nothing.
        forward_request_ids = [
            request.py_request_id
            for request in requests
            if not request.is_disagg_generation_init_state
        ]
        if forward_request_ids or not requests:
            self.prepare_sparse_batch(forward_request_ids)

    def prepare_sparse_batch(self, request_ids: list[int]) -> None:
        """Publish scheduler or warmup requests, or join an already published batch.

        Attention metadata calls this outside capture with its final request order.
        Graph padding may append dummy rows; the live request order must still agree
        with scheduling. Warmup requests can reach this without prepare_resources().
        """
        if self._sparse_batch_prepared:
            self.get_sparse_batch(request_ids)
            torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
        else:
            self._prepare_sparse_batch(request_ids)

    def _prepare_sparse_batch(self, request_ids: list[int]) -> None:
        """Publish scheduled caches in model input order before a forward.

        The allocation persists across steps and CUDA graph profiles. Membership lasts
        until the post-forward resource update: Batch temporarily owns every member's
        host page-index buffers, including the dense SWA and indexer groups. Returning
        those buffers before scheduler mutations keeps the legacy tables up to date.
        """
        if not self._enable_kv_cache_offload or not any(
            self._compress_ratios[layer] == DEEPSEEK_V4_SPARSE_RATIO for layer in self.pp_layers
        ):
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sparse Batch publication must run outside CUDA graph capture")
        if len(request_ids) > self.max_batch_size:
            raise ValueError("Scheduled requests exceed sparse Batch capacity")
        caches = {}
        padding = False
        for request_id in request_ids:
            if request_id == CUDA_GRAPH_DUMMY_REQUEST_ID:
                padding = True
                continue
            if padding:
                raise ValueError("CUDA-graph dummy requests must form a trailing suffix")
            if request_id in caches:
                raise ValueError("Sparse Batch requires unique request IDs")
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.is_active:
                raise ValueError("Sparse Batch requires active scheduled request caches")
            if cache.beam_width != 1:
                raise ValueError("Sparse Batch supports one beam per request")
            if cache.cuda_stream != self._stream.cuda_stream:
                raise RuntimeError("Sparse Batch requires the manager's cache stream")
            if not 0 <= cache.history_length <= cache.capacity:
                raise ValueError("Invalid KVCM history frontier")
            if cache.num_blocks > self.max_blocks_per_seq:
                raise ValueError("Request allocation exceeds sparse Batch page-table capacity")
            caches[request_id] = cache

        if self._sparse_batch is None:
            self._sparse_batch = Batch(
                self.impl,
                max_batch_size=self.max_batch_size,
                max_blocks_per_seq=self.max_blocks_per_seq,
            )
        batch = self._sparse_batch
        try:
            for request_id, cache in caches.items():
                batch.add_kv_cache(cache)
                self._sparse_batch_members[request_id] = cache
            stream = torch.cuda.current_stream(self._stream.device)
            # Reusing the device mirror follows the previous forward. Publish on
            # the cache stream so metadata preparation can also join it there.
            self._stream.wait_stream(stream)
            batch.publish(CudaStream(self._stream.cuda_stream))
            stream.wait_stream(self._stream)
            self._sparse_batch_prepared = True
        finally:
            if not self._sparse_batch_prepared:
                self._release_sparse_batch()

    def get_sparse_batch(self, request_ids: list[int]) -> Batch:
        """Return the published Batch after checking the attention metadata's row order.

        Rows follow context requests then generation requests. CUDA graph padding is a
        trailing suffix of free rows, whose page indices are BAD_PAGE_INDEX and whose
        watermarks are zero. Fetch inputs must use Batch.max_batch_size rows, even when
        a smaller attention profile is executing.
        """
        if self._sparse_batch is None or not self._sparse_batch_prepared:
            raise RuntimeError("Sparse Batch must be prepared before model execution")
        live_ids = list(self._sparse_batch_members)
        if (
            len(request_ids) > self._sparse_batch.max_batch_size
            or request_ids[: len(live_ids)] != live_ids
            or any(req != CUDA_GRAPH_DUMMY_REQUEST_ID for req in request_ids[len(live_ids) :])
        ):
            raise ValueError("Attention request order does not match the published sparse Batch")
        return self._sparse_batch

    def _release_sparse_batch(self) -> None:
        """Return host page-index ownership before updating or releasing request caches."""
        self._sparse_batch_prepared = False
        if self._sparse_batch is None:
            return
        self._sparse_batch.wait_published()
        for request_id, cache in list(self._sparse_batch_members.items()):
            self._sparse_batch.remove_kv_cache(cache)
            del self._sparse_batch_members[request_id]
            # Removal restores the cache's internal indices. Rebind the runtime's
            # host table so subsequent resizes also reach SWA and indexer consumers.
            self._set_page_index_bufs(request_id, cache)

    def suspend_request(self, req: llm_request.LlmRequest) -> None:
        self._release_sparse_batch()
        self._order_sparse_history_update([req])
        super().suspend_request(req)

    def free_resources(self, request: llm_request.LlmRequest, pin_on_release: bool = False) -> None:
        self._release_sparse_batch()
        self._order_sparse_history_update([request])
        super().free_resources(request, pin_on_release)

    def release_index_slot(self, request_id: int) -> None:
        self._release_sparse_batch()
        super().release_index_slot(request_id)

    def shutdown(self) -> None:
        self._release_sparse_batch()
        self._sparse_batch = None
        super().shutdown()

    def _validate_sparse_history_policy(self) -> None:
        """Check fixed history policies during attention metadata initialization."""
        if (
            self.enable_block_reuse
            or self.kv_compression_manages_history
            or self._has_cp_helix
            or self.is_draft
        ):
            raise NotImplementedError(
                "Sparse offload requires block reuse, KV compression, CP Helix, and draft caches disabled"
            )

    def _order_sparse_history_update(self, requests: Iterable[llm_request.LlmRequest]) -> None:
        """Order writeback after all model producers and attention readers.

        The executor orders its calling stream after model forward.
        Join it onto the cache stream before resize writes back completed units
        or recycles pages. prepare_sparse_batch() supplies the reverse dependency
        for the next forward.
        """
        if not self._enable_kv_cache_offload:
            return
        active = False
        for request in requests:
            cache = self.kv_cache_map.get(request.py_request_id)
            if cache is None or not cache.is_active:
                continue
            if cache.cuda_stream != self._stream.cuda_stream:
                raise RuntimeError("Sparse history updates require the manager's cache stream")
            active = True
        if active:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Sparse history updates must run outside CUDA graph capture")
            self._stream.wait_stream(torch.cuda.current_stream(self._stream.device))

    def update_context_resources(self, scheduled_batch: ScheduledRequests) -> None:
        self._release_sparse_batch()
        self._order_sparse_history_update(scheduled_batch.context_requests)
        super().update_context_resources(scheduled_batch)

    def try_allocate_generation(self, req: llm_request.LlmRequest) -> bool:
        """Reserve a decode request's next input token unless it is already reserved.

        With offload, history advances right after each forward, so at scheduling
        time a cache holding capacity beyond its history has already reserved this
        step. Attention-DP padding dummies are created that way. Growing such a
        cache again would exceed the sparse cache's one-token bound between
        capacity and history. No growth is recorded, so a skipped step reverts
        nothing.
        """
        if self._enable_kv_cache_offload:
            cache = self.kv_cache_map.get(req.py_request_id)
            if cache is not None and cache.is_active and cache.capacity > cache.history_length:
                return True
        return super().try_allocate_generation(req)

    def update_forward_resources(
        self, scheduled_batch: ScheduledRequests, *, wait_for_input_copy: Callable[[], None]
    ) -> None:
        """Advance sparse decode history from executed inputs before the next schedule.

        Offload decode validates one input token per request, with speculation
        disabled. That input's KV is produced by this forward; the sampled
        output has no KV yet. Queue writeback after the forward on the cache
        stream, independently of when the executor consumes sampling results.
        """
        if not self._enable_kv_cache_offload or not scheduled_batch.generation_requests:
            return
        wait_for_input_copy()
        self._release_sparse_batch()
        self._order_sparse_history_update(scheduled_batch.generation_requests)
        for request in scheduled_batch.generation_requests:
            request_id = request.py_request_id
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.is_active:
                continue
            history_length = cache.history_length + 1
            if history_length > cache.capacity:
                raise ValueError(f"Sparse decode request {request_id} has no reserved input token")
            if not cache.resize(None, history_length):
                raise ValueError(
                    f"Failed to advance sparse KV history for request {request_id} "
                    f"to {history_length} tokens after model forward"
                )
            self._allocated_draft_lens.pop(request_id, None)

    def update_resources(
        self,
        scheduled_batch: ScheduledRequests,
        attn_metadata: "AttentionMetadata | None" = None,
        kv_cache_dtype_byte_size: float | None = None,
    ) -> None:
        if self._enable_kv_cache_offload:
            # Sparse decode has no speculative slots to rewind. History follows
            # forward completion; delayed sampling updates must neither rewind it
            # nor detach the Batch prepared for a later forward.
            return
        super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    def get_sparse_offload_descriptors(self) -> dict[int, SparseOffloadLayerDescriptor]:
        """Describe each local ratio-4 layer's complete sparse fetch buffer set.

        KVCM orders buffers by pool index, retaining declaration order within
        each pool. Pool addresses and page-index scales describe host storage;
        they do not transform the physical arena indices returned by fetch.
        """
        if not self._enable_kv_cache_offload:
            raise ValueError("Sparse offload metadata requires an offload-enabled cache manager")
        local_layers = [
            layer
            for layer in self.pp_layers
            if self._compress_ratios[layer] == DEEPSEEK_V4_SPARSE_RATIO
        ]
        if not local_layers:
            return {}
        if self.use_fp8_ds_mla:
            raise NotImplementedError(
                "Sparse fp8_ds_mla compressed KV requires a selection layout with "
                "separate data and footer-scale regions"
            )

        # Coalesced-buffer positions are the pool indices within each group.
        pool_indices = {
            buffer_id: pool_index
            for group in self.impl.pool_group_descs
            for variant in group.slot_desc.variants
            for pool_index, coalesced in enumerate(variant.coalesced_buffers)
            for buffer_id in coalesced.buffer_ids
        }
        layer_configs = {
            layer.layer_id: layer
            for layer in self.impl.init_config.layers
            if isinstance(layer, AttentionLayerConfig)
        }
        # Match KVCM's arena walk: configuration layer order, then declaration
        # order within each pool. Host coalesced-buffer order is not used here.
        sparse_pool_buffers: dict[tuple[LayerGroupId, int], list[BufferId]] = defaultdict(list)
        max_units_by_group: dict[LayerGroupId, int] = defaultdict(int)
        for layer_config in layer_configs.values():
            cache = layer_config.sparse_cache
            if cache is None:
                continue
            group_id = self.impl.get_layer_group_id(layer_config.layer_id)
            for buffer in layer_config.buffers:
                if buffer.is_sparse:
                    buffer_id = BufferId(layer_config.layer_id, buffer.role)
                    sparse_pool_buffers[group_id, pool_indices[buffer_id]].append(buffer_id)
                    max_units_by_group[group_id] = max(
                        max_units_by_group[group_id], cache.max_cached_units
                    )
        arena_layouts = {
            buffer_id: (len(buffers), position)
            for buffers in sparse_pool_buffers.values()
            for position, buffer_id in enumerate(buffers)
        }
        max_leases = max(len(batch.kv_caches) for batch in self.impl.init_config.constraints)
        descriptors = {}
        for layer in local_layers:
            layer_id = self._layer_attn_to_layer_id[layer, DeepseekV4AttentionType.COMPRESS]
            role = DeepseekV4AttentionType.COMPRESS.role
            if not self.impl.is_sparse(layer_id, role):
                raise ValueError(f"Layer {layer}'s COMPRESS buffer must be sparse")
            units_per_page = self.impl.get_sparse_cache_units_per_page(layer_id)
            if (
                units_per_page <= 0
                or self.tokens_per_block != units_per_page * DEEPSEEK_V4_SPARSE_RATIO
            ):
                raise ValueError("Sparse offload requires one compressed row per selection unit")
            # A layer holds the compressed KV beside the indexer that selects over it, so
            # take only the sparse buffers: the dense ones are not fetched through the cache.
            cache = layer_configs[layer_id].sparse_cache
            if (
                cache is None
                or cache.max_cached_units <= 0
                or (cache.num_selected_blocks != self._sparse_offload_topk)
            ):
                raise ValueError(
                    "A sparse layer must configure the runtime's selection width and cache capacity"
                )
            buffers = [
                BufferId(layer_id, buffer.role)
                for buffer in layer_configs[layer_id].buffers
                if buffer.is_sparse
            ]
            if BufferId(layer_id, role) not in buffers:
                raise ValueError(f"Layer {layer}'s sparse buffers must include COMPRESS")
            buffers.sort(key=pool_indices.__getitem__)
            scale_id = BufferId(layer_id, COMPRESS_BLOCK_SCALE_ROLE)
            if self._use_nvfp4_compress and scale_id not in buffers:
                raise ValueError("Sparse NVFP4 COMPRESS requires a block-scale buffer")
            if self._use_nvfp4_compress:
                data_layout = arena_layouts[BufferId(layer_id, role)]
                scale_layout = arena_layouts[scale_id]
                if data_layout != scale_layout:
                    raise ValueError(
                        "Sparse NVFP4 COMPRESS data and scale arenas must have matching "
                        "buffer counts and positions, including padding, to share physical "
                        f"indices; got {data_layout} and {scale_layout}."
                    )
            # A sparse buffer's size is a whole page; its arena region holds units.
            unit_bytes = {
                buffer.role: buffer.size // units_per_page
                for buffer in layer_configs[layer_id].buffers
            }
            group_id = self.impl.get_layer_group_id(layer_id)
            reserved_units = (
                (max_units_by_group[group_id] + units_per_page - 1)
                // units_per_page
                * units_per_page
            )
            arena_bases = tuple(
                self.impl.get_sparse_cache_base_address(buffer.layer_id, buffer.role)
                for buffer in buffers
            )
            padding_regions = []
            for buffer, base in zip(buffers, arena_bases):
                if buffer.role == role or buffer == scale_id:
                    continue
                pool_count, position = arena_layouts[buffer]
                buffer_bytes = reserved_units * unit_bytes[buffer.role]
                padding_regions.append(
                    SparseOffloadPaddingRegion(
                        base_address=base + position * buffer_bytes,
                        bytes_per_lease=buffer_bytes,
                        lease_stride=pool_count * buffer_bytes,
                    )
                )
            descriptors[layer] = SparseOffloadLayerDescriptor(
                buffer_ids=tuple(buffers),
                arena_base_addresses=arena_bases,
                arena_row_counts=tuple(
                    max_leases * arena_layouts[buffer][0] * reserved_units for buffer in buffers
                ),
                data_buffer_index=buffers.index(BufferId(layer_id, role)),
                scale_buffer_index=buffers.index(scale_id) if scale_id in buffers else None,
                group_id=group_id,
                unit_tokens=DEEPSEEK_V4_SPARSE_RATIO,
                padding_regions=tuple(padding_regions),
            )

        return descriptors

    def _initialize_sparse_arena_buffers(
        self, layers: dict[int, SparseOffloadLayerDescriptor]
    ) -> None:
        """Cache tensor views over KVCM-owned arena regions before graph capture.

        Each view is [pool_rows, 1, row_elements], so an arena index is also a
        one-row page index. Pool rows cover all leases and buffers, including
        padding roles. Layers sharing a region reuse the same tensor storage.
        KVCM must keep the allocation alive while these views or graphs use it.
        """
        if not layers:
            return
        with torch.cuda.device(self._stream.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Sparse arena views must be initialized before graph capture")
            pool_views: dict[int, torch.Tensor] = {}
            for layer, descriptor in layers.items():
                specs = [
                    (
                        descriptor.data_buffer_index,
                        DataType.UINT8 if self._use_nvfp4_compress else self.dtype,
                        (self.head_dim + self.nvfp4_residual_dim) // 2
                        if self._use_nvfp4_compress
                        else self.head_dim,
                    )
                ]
                if descriptor.scale_buffer_index is not None:
                    specs.append(
                        (
                            descriptor.scale_buffer_index,
                            DataType.FP8,
                            (self.head_dim + self.nvfp4_residual_dim) // NVFP4_VECTOR_SIZE,
                        )
                    )
                views = []
                for index, dtype, row_elements in specs:
                    base = descriptor.arena_base_addresses[index]
                    if base not in pool_views:
                        pool_views[base] = convert_to_torch_tensor(
                            TensorWrapper(
                                base, dtype, (descriptor.arena_row_counts[index], 1, row_elements)
                            )
                        )
                    views.append(pool_views[base])
                self._sparse_arena_buffers[layer] = (views[0], views[1] if len(views) > 1 else None)

    def get_sparse_compress_buffers(
        self, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return cached data/scales arena views indexed by prepared or fetched rows."""
        return self._sparse_arena_buffers[layer_idx]

    def _initialize_sparse_arena_padding(
        self, layers: dict[int, SparseOffloadLayerDescriptor]
    ) -> None:
        """Initialize unused roles once, before requests can acquire arena leases.

        The arena's per-pool lease strides follow its configuration, independently
        of host page capacities. Views cover only padding buffers; real KV and
        scale buffers are initialized by their producers. KVCM owns these allocations.
        """
        regions = [region for layer in layers.values() for region in layer.padding_regions]
        if not regions:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Sparse arena padding must be initialized before graph capture")
        max_leases = max(len(batch.kv_caches) for batch in self.impl.init_config.constraints)
        with torch.cuda.device(self._stream.device), torch.cuda.stream(self._stream):
            for region in regions:
                convert_to_torch_tensor(
                    TensorWrapper(
                        region.base_address,
                        torch.uint8,
                        (max_leases, region.bytes_per_lease),
                        strides=(region.lease_stride, 1),
                    )
                ).zero_()
        # Construction completes before any producer or captured graph can use the arena.
        self._stream.synchronize()

    @nvtx_range_debug("dsv4_prepare_sparse_offload")
    def prepare_sparse_offload(
        self,
        state: SparseOffloadState,
        request_ids: list[int],
        *,
        beam_width: int,
    ) -> None:
        """Bind the published Batch and refresh the active count outside capture.

        Scheduling publishes the page tables and watermarks in request order.
        The native Batch and its device views persist across steps and graph
        profiles; only trailing dummy requests may extend the scheduled rows.
        This prepares fetch metadata, not compressor write destinations.
        """
        state.prepared = False
        state.write_layer = None
        if not self._enable_kv_cache_offload or beam_width != 1:
            raise ValueError("Sparse offload preparation requires offload enabled and beam width 1")
        with torch.cuda.device(state.selected_history_units.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Sparse offload metadata must be prepared outside graph capture")
            batch = self.get_sparse_batch(request_ids)
            if batch.max_batch_size != state.selected_history_units.shape[0]:
                raise ValueError("Sparse fetch buffers must use the native Batch capacity")
            if state.batch is not None and state.batch is not batch:
                raise RuntimeError(
                    "Sparse Batch allocation must remain stable across graph replays"
                )
            torch.cuda.current_stream().wait_stream(self._stream)
            if state.batch is None:
                state.history_lengths = torch.from_dlpack(batch.history_lengths())
                state.batch = batch
            state.active_request_count.fill_(len(self._sparse_batch_members))
            state.selected_history_units.fill_(BAD_PAGE_INDEX)
            for out in state.fetched_indices.values():
                out.fill_(BAD_PAGE_INDEX)
            for out in state.write_indices.values():
                out.fill_(BAD_PAGE_INDEX)
        state.prepared = True

    def prepare_sparse_compressor_write_indices(
        self, state: SparseOffloadState, layer_idx: int
    ) -> torch.Tensor:
        """Reserve this layer's pinned destinations on the compressor's stream.

        Return int32[B, F] in Batch and descriptor buffer order. The caller
        must join Batch publication before this call, and join the compressor
        stream before fetch/attention or a history update. No KV is fetched here.
        Padding roles were initialized during manager construction. The fused
        compressor uses the data column as a one-row page table for data and
        NVFP4 scales, whose matching geometry is checked during descriptor construction
        and, once per layer outside graph capture, against the indices native returns.
        Each buffer still uses its own arena base and byte stride.
        """
        if (
            not self._enable_kv_cache_offload
            or not state.prepared
            or state.is_prefill
            or state.batch is None
        ):
            raise RuntimeError("Sparse compressor writes require prepared decode metadata")
        descriptor = state.layers[layer_idx]
        out = state.write_indices[len(descriptor.buffer_ids)]
        state.write_layer = None
        with torch.cuda.device(out.device):
            self.impl.prepare_sparse_write_indices(
                batch=state.batch,
                buffers=descriptor.buffer_ids,
                out=out,
                stream=CudaStream(torch.cuda.current_stream().cuda_stream),
            )
            if (
                descriptor.scale_buffer_index is not None
                and layer_idx not in self._paired_write_layers
                and not torch.cuda.is_current_stream_capturing()
            ):
                self._check_paired_write_indices(out, descriptor, layer_idx)
        state.write_layer = layer_idx
        return out

    def _check_paired_write_indices(
        self, out: torch.Tensor, descriptor: SparseOffloadLayerDescriptor, layer_idx: int
    ) -> None:
        """Check that native gives a layer's data and block scales the same write rows.

        The fused compressor and the NVFP4 gather address scales with the data
        column. Descriptor construction re-derives the arena layout to check
        that, so it cannot see a native change; the indices themselves can. A
        layer counts as checked once a live row has been compared. Reading the
        result waits for the stream, so this runs only until then.
        """
        data = out[:, descriptor.data_buffer_index]
        scale = out[:, descriptor.scale_buffer_index]
        differs, live = torch.stack(((data != scale).any(), (data >= 0).any())).tolist()
        if differs:
            raise RuntimeError(
                f"Sparse NVFP4 layer {layer_idx}: the KV cache manager wrote data and block "
                "scales to different arena rows, but the compressor and attention address "
                "scales with the data row. Its sparse arena no longer pairs the two buffers."
            )
        if live:
            self._paired_write_layers = self._paired_write_layers | {layer_idx}

    @nvtx_range_debug("dsv4_fetch_sparse_indices")
    def fetch_sparse_indices(
        self,
        state: SparseOffloadState,
        layer_idx: int,
        topk_indices: torch.Tensor,
        request_rows: torch.Tensor,
        compressed_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Fetch one layer's history into compact per-buffer arena indices.

        Inputs are CUDA int32: top-k compressed-row ordinals [Q, K], query-to-
        Batch rows [Q], and compressed lengths [R] in scheduler/request order.
        Live query rows are a permutation of the active Batch prefix, one query
        per request. Q and R may be smaller than Batch.max_batch_size.

        Return int32[B, F, K] in Batch row order and descriptor.buffer_ids order.
        Each index is relative to that buffer's sparse arena base. Top-k slots
        keep their order; padding and in-flight selections produce -1. Attention
        must supply the in-flight write index separately, without treating this
        output as an ordinal-indexed page table or rescaling its indices.

        The caller joins indexer/compressor producers before this call. Selection,
        fetch and the consuming attention stay ordered on the current stream.
        Output storage is reused by subsequent layers after attention consumes it.
        """
        if (
            not self._enable_kv_cache_offload
            or not state.prepared
            or state.is_prefill
            or state.batch is None
            or state.history_lengths is None
        ):
            raise RuntimeError("Sparse fetch requires prepared decode metadata")
        descriptor = state.layers[layer_idx]
        with torch.cuda.device(state.selected_history_units.device):
            select_sparse_history_units(
                topk_indices,
                request_rows,
                state.active_request_count,
                compressed_lengths,
                state.history_lengths,
                state.selected_history_units,
                unit_tokens=descriptor.unit_tokens,
            )
            out = state.fetched_indices[len(descriptor.buffer_ids)]
            # Fetch even for an empty selection so every output slot is refreshed
            # on graph replay. Native Batch supplies page tables and watermarks.
            self.impl.fetch_sparse_pages(
                batch=state.batch,
                buffers=descriptor.buffer_ids,
                selected=state.selected_history_units,
                out=out,
                stream=CudaStream(torch.cuda.current_stream().cuda_stream),
            )
        return out

    @nvtx_range_debug("dsv4_copy_batch_compress_block_tables")
    def copy_batch_compress_block_tables(
        self,
        dst_tensor: torch.Tensor,
        request_ids: List[int],
        compress_ratio: int,
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the COMPRESS block table for one compression ratio and copy it to the destination."""
        if self._enable_kv_cache_offload and compress_ratio == DEEPSEEK_V4_SPARSE_RATIO:
            raise RuntimeError("Sparse COMPRESS requires per-buffer arena indices")
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        staging = self._host_compress_block_tables_staging[compress_ratio]
        if compress_ratio == 4:
            pool_id = self._csa_compress_pool_id
            scale = self._csa_compress_scale
        elif compress_ratio == 128:
            pool_id = self._hca_compress_pool_id
            scale = self._hca_compress_scale
        else:
            raise ValueError(
                f"Unsupported compress ratio {compress_ratio} for copy_batch_compress_block_tables"
            )

        if pool_id is None or scale is None:
            raise RuntimeError(
                f"Missing COMPRESS pool metadata for compress ratio {compress_ratio}"
            )
        staging[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)
        dst_tensor[:num_seqs].copy_(staging[:num_seqs], non_blocking=True)

    @nvtx_range_debug("dsv4_copy_batch_indexer_compress_block_tables")
    def copy_batch_indexer_compress_block_tables(
        self,
        host_block_table: torch.Tensor,
        request_ids: List[int],
        beam_width: int,
        num_contexts: int,
        num_seqs: int,
    ) -> None:
        """Build the shared INDEXER_COMPRESS compatibility block table."""
        assert beam_width == 1, "DSV4 only supports beam width 1 now"
        copy_idx = self.index_mapper.get_copy_index(request_ids, num_contexts, beam_width)
        pool_id = self._csa_indexer_compress_pool_id
        scale = self._csa_indexer_compress_scale
        if pool_id is None or scale is None:
            raise RuntimeError("Missing INDEXER_COMPRESS pool metadata")
        host_block_table[:num_seqs] = self._compute_shared_block_table(pool_id, scale, copy_idx)

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig, mapping: Mapping, **kwargs):
        config = model_config.pretrained_config
        head_dim = config.kv_lora_rank + config.qk_rope_head_dim
        index_head_dim = model_config.sparse_attention_config.index_head_dim
        pp_layers = mapping.pp_layers(model_config.get_num_attention_layers())
        compress_ratios = [
            model_config.sparse_attention_config.compress_ratios[layer] for layer in pp_layers
        ]
        quant_config = model_config.quant_config
        has_nvfp4_compress = bool(
            quant_config is not None and quant_config.quant_mode.has_fp4_kv_cache()
        )
        has_fp8_kv_cache = bool(
            quant_config is not None
            and (quant_config.quant_mode.has_fp8_kv_cache() or has_nvfp4_compress)
        )
        indexer_k_dtype = model_config.sparse_attention_config.indexer_k_dtype
        kv_cache_config = kwargs.get("kv_cache_config")
        use_fp8_ds_mla = (
            kv_cache_config is not None
            and getattr(kv_cache_config, "dtype", "auto") == "fp8_ds_mla"
        )
        offload = model_config.sparse_attention_config.enable_kv_cache_offload
        non_sliding_attn_size_per_token = _estimate_non_sliding_attn_size_per_token(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            indexer_k_dtype=indexer_k_dtype,
            use_fp8_ds_mla=use_fp8_ds_mla,
            has_nvfp4_compress=has_nvfp4_compress,
            offload_sparse_history=offload,
        )
        generation_capacity_headroom = None
        if offload:
            _, generation_capacity_headroom = _get_generation_kv_capacity(
                kwargs.get("spec_config"), is_draft=kwargs.get("is_draft", False)
            )
        swa_size_per_token, swa_size_per_request = _estimate_swa_cache_size(
            head_dim,
            index_head_dim,
            compress_ratios,
            has_fp8_kv_cache,
            kwargs["tokens_per_block"],
            model_config.sparse_attention_config.window_size,
            context=False,
            scratch=False,
            indexer_k_dtype=indexer_k_dtype,
            use_fp8_ds_mla=use_fp8_ds_mla,
            generation_capacity_headroom=generation_capacity_headroom,
        )
        max_batch_size = int(kwargs.get("max_batch_size") or 0)
        return (
            ceil(non_sliding_attn_size_per_token + swa_size_per_token),
            swa_size_per_request * max_batch_size,
        )

    def _iter_guard_candidate_buffers(
        self,
    ) -> Iterable[Tuple[int, DeepseekV4AttentionType, torch.Tensor]]:
        """Guard-page candidates for DeepSeek-V4's multi-attn-type layers.

        The base hook calls ``get_buffers(layer_idx)``, but this subclass's
        ``get_buffers`` requires an ``attn_type``, so the positional base call
        would raise ``TypeError`` while reserving the guard page. Iterate the
        (layer, attn_type) map instead and yield the ``attn_type`` too: the
        guard page-index lookup needs it (see ``_resolve_guard_candidate``)
        because DSV4 allocates no generic ``Role.KEY`` buffer. De-duplicate
        physical buffers the same way the invalid-value check does.
        """
        buffers_handled = set()
        for (layer, attn), layer_id in self._layer_attn_to_layer_id.items():
            buffer_key = (layer_id, attn.role)
            if buffer_key in buffers_handled:
                continue
            buffers_handled.add(buffer_key)
            buffer = self.get_buffers(layer, attn)
            if buffer is None:
                continue
            yield layer, attn, buffer

    def _resolve_guard_candidate(
        self, candidate: Tuple[int, DeepseekV4AttentionType, torch.Tensor]
    ) -> Tuple[Tuple[int, DataRole], torch.Tensor, List[int]]:
        """Resolve a DeepSeek-V4 guard candidate to ``(key, buffer, pages)``.

        The base path resolves the page index through
        ``get_batch_cache_indices(layer_idx)`` ->
        ``get_page_index_scale(layer_offsets[layer_idx], Role.KEY)``, but DSV4
        allocates no generic ``Role.KEY`` buffer and keys offsets/pools per
        (layer, attn_type), so that lookup raises "Unknown buffer id". Use the
        attn-specific ``get_cache_indices`` instead, which resolves the pool,
        converter and page-index mode from ``_layer_attn_to_layer_id`` and
        ``attn_type.role``.

        Key by ``(layer_id, attn_type.role)`` -- the physical-buffer identity
        the candidate iterator de-duplicates on -- so two distinct attn buffers
        on the SAME model layer do not clobber each other's guard page in
        ``_guard_page_by_layer`` (a bare ``layer_idx`` key would).
        """
        layer, attn, buffer = candidate
        pages = self.get_cache_indices(_GUARD_PAGE_REQUEST_ID, layer, attn)
        layer_id = self._layer_attn_to_layer_id[(layer, attn)]
        return (layer_id, attn.role), buffer, pages

    def _guard_reservation_detail(self) -> str:
        # Report the DSV4-specific shape: guard pages are keyed per physical
        # (layer_id, role) buffer, and one model layer can host several such
        # buffers -- the clobber case the per-buffer key exists to avoid.
        buffers_per_layer: Dict[int, set] = defaultdict(set)
        for layer, attn in self._layer_attn_to_layer_id:
            buffers_per_layer[layer].add((self._layer_attn_to_layer_id[(layer, attn)], attn.role))
        multi = sum(1 for bufs in buffers_per_layer.values() if len(bufs) > 1)
        return (
            f"({len(self._guard_page_by_layer)} attn buffers guarded across "
            f"{len(buffers_per_layer)} model layers; {multi} model layers host "
            f">1 attn buffer)"
        )

    def _iter_cache_cleanup_buffers(self) -> Iterable[tuple[torch.Tensor, int | None]]:
        """Yield device storage and arena views with their optional storage guard-page index.

        Host-backed sparse history is left out. Nothing reads a row of it before
        a transfer or a write-back fills it, and walking it would cross PCIe for
        every page of the host budget.
        """
        buffers_handled = set()

        # Handle each attention buffer from start to end to traverse the whole
        # KV cache. Multiple attention buffers can now share one cache layer.
        for (layer, attn), layer_id in self._layer_attn_to_layer_id.items():
            buffers = [((layer_id, attn.role), self.get_buffers(layer, attn))]
            if self._use_nvfp4_compress and attn == DeepseekV4AttentionType.COMPRESS:
                buffers.append(
                    (
                        (layer_id, COMPRESS_BLOCK_SCALE_ROLE),
                        self.get_compress_scale_buffers(layer),
                    )
                )
            for buffer_key, buffer in buffers:
                if buffer_key in buffers_handled:
                    continue
                buffers_handled.add(buffer_key)
                if self._enable_kv_cache_offload and self.impl.is_sparse(*buffer_key):
                    continue
                yield buffer, self._guard_page_by_layer.get(buffer_key)

        # Arena views span complete GPU pools and can be shared by several layers.
        # Their physical row indices are independent of the host storage guard pages.
        arenas_handled = set()
        for arenas in self._sparse_arena_buffers.values():
            for arena in arenas:
                if arena is None or arena.data_ptr() in arenas_handled:
                    continue
                arenas_handled.add(arena.data_ptr())
                yield arena, None

    @torch.inference_mode()
    def check_invalid_values_in_kv_cache(self, fill_with_zero: bool = False) -> bool:
        if self._enable_kv_cache_offload:
            # Finish cache-stream writeback, which reads the arenas, before
            # scanning or clearing them.
            torch.cuda.current_stream(self._stream.device).wait_stream(self._stream)
        some_checks_unavailable = False
        has_invalid_values = torch.tensor(
            [False], dtype=torch.bool, device=torch.cuda.current_device()
        )
        for buffer, guard_page in self._iter_cache_cleanup_buffers():
            # Process in chunks of 256 pages (or sparse rows) to avoid OoM.
            for i in range(0, buffer.shape[0], 256):
                buffer_slice = buffer[i : i + 256]
                try:
                    has_invalid_values.logical_or_(torch.isnan(buffer_slice).any())
                    has_invalid_values.logical_or_(torch.isinf(buffer_slice).any())
                except NotImplementedError:
                    some_checks_unavailable = True
            if fill_with_zero:
                buffer.zero_()
                if guard_page is not None and self._guard_page_value is not None:
                    _fill_kv_pages(buffer, [guard_page], self._guard_page_value)
        torch.cuda.synchronize()

        if some_checks_unavailable:
            logger.warning(
                "`torch.isnan` or `torch.isinf` is not implemented for current kv cache dtype, "
                "related checks are skipped"
            )
        return bool(has_invalid_values)
