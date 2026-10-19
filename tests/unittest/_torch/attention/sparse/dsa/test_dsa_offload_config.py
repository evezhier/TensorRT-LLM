# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSA sparse KV cache offload configuration and its lowering."""

import pytest
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    DeepSeekSparseAttentionConfig,
    DeepSeekV4SparseAttentionConfig,
    SparseKvCacheOffloadConfig,
)

_OFFLOAD = {"gpu_cache_factor": 3.0, "host_history_size": 2 * 1024**3}


def _glm_config() -> PretrainedConfig:
    return PretrainedConfig(
        architectures=["GlmMoeDsaForCausalLM"],
        num_hidden_layers=7,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_rope_interleave=True,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_dsa_offload_lowers_into_sparse_params(enabled):
    config = DeepSeekSparseAttentionConfig(
        index_topk=2048,
        enable_kv_cache_offload=enabled,
        kv_cache_offload_config=_OFFLOAD if enabled else {},
    )
    expected = SparseKvCacheOffloadConfig(**_OFFLOAD) if enabled else None
    assert config.get_kv_cache_offload_config() == expected
    assert config.to_sparse_params().enable_kv_cache_offload is enabled
    assert config.to_sparse_metadata_params().enable_kv_cache_offload is enabled


def test_dsa_offload_sizing_requires_offload():
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        DeepSeekSparseAttentionConfig(kv_cache_offload_config=_OFFLOAD)


def test_dsa_offload_defers_the_top_k_check_to_the_checkpoint():
    # index_topk comes from the checkpoint when the user leaves it unset.
    assert DeepSeekSparseAttentionConfig(enable_kv_cache_offload=True).index_topk is None
    with pytest.raises(ValueError, match="positive index_topk"):
        DeepSeekSparseAttentionConfig(enable_kv_cache_offload=True, index_topk=0)


def test_deepseek_v4_offload_keeps_its_ratio_check():
    with pytest.raises(ValueError, match="ratio-4"):
        DeepSeekV4SparseAttentionConfig(compress_ratios=[1, 128], enable_kv_cache_offload=True)
    with pytest.raises(ValueError, match="requires enable_kv_cache_offload"):
        DeepSeekV4SparseAttentionConfig(kv_cache_offload_config=_OFFLOAD)


@pytest.mark.parametrize("enabled", [False, True])
def test_glm_model_config_preserves_sparse_offload(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(
        "tensorrt_llm._torch.model_config.load_pretrained_config",
        lambda *args, **kwargs: _glm_config(),
    )
    sparse_config = DeepSeekSparseAttentionConfig(
        enable_kv_cache_offload=enabled,
        kv_cache_offload_config=_OFFLOAD if enabled else {},
    )

    resolved = ModelConfig.from_pretrained(
        str(tmp_path),
        sparse_attention_config=sparse_config,
        attn_backend="TRTLLM",
        moe_backend="TRTLLM",
    ).sparse_attention_config

    assert resolved.enable_kv_cache_offload is enabled
    assert resolved.kv_cache_offload_config == sparse_config.kv_cache_offload_config
    # The checkpoint fills the selection width the offload cache is sized by.
    assert resolved.index_topk == 2048
