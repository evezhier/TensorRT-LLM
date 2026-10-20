# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSACacheManagerV2 with sparse KV cache offload, on the native KV cache manager."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.sparse.offload import allocate_sparse_offload_state
from tensorrt_llm._torch.attention.backends.sparse.offload_kernels import merge_sparse_arena_indices
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import Role
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, get_sm_version
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 100,
    reason="DSA KV cache offload needs an SM100+ GPU",
)

NUM_LAYERS = 7
FULL_INDEXER_LAYERS = (0, 1, 2, 6)
TOPK = 64
TOKENS_PER_BLOCK = 64
MAX_BATCH_SIZE = 4
HEAD_DIM = 576
INDEXER_BYTES_PER_TOKEN = 132


def _pretrained_config(num_layers: int = NUM_LAYERS) -> SimpleNamespace:
    # GLM-5.2's indexer pattern: a full layer every fourth from layer 3 on.
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk=TOPK,
        index_head_dim=128,
        index_n_heads=32,
        qk_rope_head_dim=64,
    )


def _manager(offload: bool = True, num_layers: int = NUM_LAYERS) -> DSACacheManagerV2:
    sparse_config = DeepSeekSparseAttentionConfig(
        index_head_dim=128,
        index_n_heads=32,
        index_topk=TOPK,
        enable_kv_cache_offload=offload,
        kv_cache_offload_config=(
            {"gpu_cache_factor": 2.0, "host_history_size": 1 << 30} if offload else {}
        ),
    )
    return DSACacheManagerV2(
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, max_tokens=4096, host_cache_size=0),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=1024,
        max_batch_size=MAX_BATCH_SIZE,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        sparse_attention_config=sparse_config,
        pretrained_config=_pretrained_config(num_layers),
    )


def _request(request_id: int, prompt_len: int) -> LlmRequest:
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=32,
        input_tokens=list(range(prompt_len)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


@pytest.fixture
def manager():
    manager = _manager()
    yield manager
    manager.shutdown()


def test_offload_puts_every_latent_in_host_memory_and_keeps_the_indexer(manager):
    layers = manager.kv_cache_manager_py_config.layers
    assert len(layers) == NUM_LAYERS
    for layer in layers:
        roles = {str(buffer.role): buffer for buffer in layer.buffers}
        assert roles[str(Role.KEY)].is_sparse
        assert layer.sparse_cache.num_selected_blocks == TOPK
        assert layer.sparse_cache.unit_tokens == 1
        has_indexer = str(Role.INDEX_KEY) in roles
        assert has_indexer == (int(layer.layer_id) in FULL_INDEXER_LAYERS)
        if has_indexer:
            assert not roles[str(Role.INDEX_KEY)].is_sparse
    for local_layer in range(NUM_LAYERS):
        expected = INDEXER_BYTES_PER_TOKEN if local_layer in FULL_INDEXER_LAYERS else 0
        assert manager.get_layer_bytes_per_token(local_layer, Role.ALL) == expected
        assert manager.get_layer_bytes_per_token(local_layer, Role.KEY) == HEAD_DIM
    descriptors = manager.get_sparse_offload_descriptors()
    assert sorted(descriptors) == list(range(NUM_LAYERS))
    for layer, descriptor in descriptors.items():
        assert descriptor.unit_tokens == 1
        assert descriptor.scale_buffer_index is None
        assert len(descriptor.buffer_ids) == 1
        pool_pointers, pool_mapping = manager.get_sparse_latent_write_pool(layer)
        assert pool_pointers.tolist() == [[descriptor.arena_base_addresses[0], 0]]
        assert pool_mapping.tolist() == [[0, 0]]


def test_offload_refuses_the_unified_pool_view(manager):
    with pytest.raises(RuntimeError, match="host memory"):
        manager.get_unique_primary_pool()


def test_decode_step_writes_fetches_and_advances_history(manager):
    prompt_len = 3 * TOKENS_PER_BLOCK + 5
    request = _request(7, prompt_len)
    assert manager.prepare_disagg_gen_init(request)
    assert manager.try_allocate_generation(request)
    cache = manager.kv_cache_map[request.py_request_id]
    assert (cache.history_length, cache.capacity) == (prompt_len, prompt_len + 1)

    descriptors = manager.get_sparse_offload_descriptors()
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=TOPK,
    )
    manager.prepare_sparse_batch([request.py_request_id])
    manager.prepare_sparse_offload(state, [request.py_request_id], beam_width=1)
    assert state.history_lengths[0].item() == prompt_len

    # The last history token and the token written this step, then older history.
    topk = torch.full((1, TOPK), -1, dtype=torch.int32, device="cuda")
    topk[0, : TOPK - 2] = torch.arange(TOPK - 2, dtype=torch.int32)
    topk[0, TOPK - 2] = prompt_len - 1
    topk[0, TOPK - 1] = prompt_len
    request_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    kv_lengths = torch.tensor([prompt_len + 1], dtype=torch.int32, device="cuda")
    out = torch.empty((1, TOPK), dtype=torch.int32, device="cuda")
    assert state.batch_rows.tolist() == [0] + [-1] * (MAX_BATCH_SIZE - 1)
    for layer in range(NUM_LAYERS):
        writes = manager.prepare_sparse_write_indices(state, layer)
        assert writes[0, 0].item() >= 0
        assert (writes[1:, 0] == -1).all()
        fetched = manager.fetch_sparse_indices(state, layer, topk, request_indices, kv_lengths)
        merge_sparse_arena_indices(
            topk,
            request_indices,
            state.batch_rows,
            state.active_request_count,
            kv_lengths,
            state.history_lengths,
            fetched[:, 0, :],
            writes[:, 0],
            out,
            unit_tokens=1,
        )
        merged = out[0].tolist()
        assert all(index >= 0 for index in merged)
        # Every selected token has its own arena row, the written one included.
        assert len(set(merged)) == TOPK
        assert merged[-1] == writes[0, 0].item()
        assert fetched[0, 0, -1].item() == -1
        rows_in_arena = descriptors[layer].arena_row_counts[0]
        assert max(merged) < rows_in_arena

    scheduled = SimpleNamespace(context_requests=[], generation_requests=[request])
    manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
    torch.cuda.synchronize()
    assert cache.history_length == prompt_len + 1
    manager.free_resources(request)


def _arena_rows(descriptor) -> torch.Tensor:
    return convert_to_torch_tensor(
        TensorWrapper(
            descriptor.arena_base_addresses[0],
            torch.int32,
            (descriptor.arena_row_counts[0], HEAD_DIM // 4),
        )
    )


@pytest.mark.parametrize(
    ("num_layers", "num_requests"), [(1, 1), (2, 1), (NUM_LAYERS, 1), (NUM_LAYERS, 3)]
)
@pytest.mark.parametrize("seed", [0, 1])
def test_fetched_rows_hold_written_history_across_evictions(seed, num_layers, num_requests):
    """Every step writes each request's token row; later fetches must return it.

    The GPU cache holds a few selections, so a long decode evicts rows and fetches
    them back from the host pages the write-back filled. Requests join at different
    steps, so the Batch's rows carry different histories. Input runs newest first,
    as on a disaggregated decode worker, while each request keeps the Batch row it
    joined in, so input positions and Batch rows disagree.
    """
    generator = torch.Generator().manual_seed(seed)
    manager = _manager(num_layers=num_layers)
    requests = [_request(9 + index, 1) for index in range(num_requests)]
    join_step = {request.py_request_id: 40 * index for index, request in enumerate(requests)}
    descriptors = manager.get_sparse_offload_descriptors()
    arenas = {layer: _arena_rows(descriptor) for layer, descriptor in descriptors.items()}
    state = allocate_sparse_offload_state(
        lambda name, shape, dtype: torch.empty(shape, dtype=dtype, device="cuda"),
        descriptors,
        max_batch_size=MAX_BATCH_SIZE,
        topk=TOPK,
    )
    active: list[LlmRequest] = []
    permuted = False
    for step in range(6 * TOKENS_PER_BLOCK):
        manager.prepare_resources(SimpleNamespace(context_requests=[], generation_requests=[]))
        for request in requests:
            if join_step[request.py_request_id] == step:
                assert manager.prepare_disagg_gen_init(request)
                active.append(request)
        for request in active:
            assert manager.try_allocate_generation(request)
        scheduled = SimpleNamespace(context_requests=[], generation_requests=active[::-1])
        manager.prepare_resources(scheduled)
        request_ids = [request.py_request_id for request in scheduled.generation_requests]
        manager.prepare_sparse_offload(state, request_ids, beam_width=1)
        # The rows the manager recorded at add_kv_cache are still the native ones.
        for member in manager._sparse_batch_members.values():
            assert manager._sparse_batch.row_of(member.cache) == member.row
        batch_rows = state.batch_rows.tolist()[: len(active)]
        permuted |= batch_rows != list(range(len(active)))
        histories = [manager.kv_cache_map[rid].history_length for rid in request_ids]
        topk = torch.full((len(active), TOPK), -1, dtype=torch.int32)
        for position, history in enumerate(histories):
            # Unit 0 came with the prompt and was never written; select from the rest,
            # always including the token written this step.
            candidates = torch.randperm(history, generator=generator)[: TOPK - 1]
            candidates = candidates[candidates > 0][: TOPK - 1].to(torch.int32)
            topk[position, : candidates.numel()] = candidates
            topk[position, -1] = history
        topk = topk.cuda()
        request_indices = torch.arange(len(active), dtype=torch.int32, device="cuda")
        kv_lengths = torch.tensor([h + 1 for h in histories], dtype=torch.int32, device="cuda")
        out = torch.empty((len(active), TOPK), dtype=torch.int32, device="cuda")
        for layer in range(num_layers):
            writes = manager.prepare_sparse_write_indices(state, layer)
            for row, rid, history in zip(batch_rows, request_ids, histories):
                write_row = writes[row, 0].item()
                assert write_row >= 0, (rid, history, layer)
                # The token's row: its position, its layer, its request.
                arenas[layer][write_row, :3] = torch.tensor(
                    [history, layer, rid], dtype=torch.int32, device="cuda"
                )
            free_rows = [row for row in range(MAX_BATCH_SIZE) if row not in batch_rows]
            assert (writes[free_rows, 0] == -1).all()
            fetched = manager.fetch_sparse_indices(state, layer, topk, request_indices, kv_lengths)
            merge_sparse_arena_indices(
                topk,
                request_indices,
                state.batch_rows,
                state.active_request_count,
                kv_lengths,
                state.history_lengths,
                fetched[:, 0, :],
                writes[:, 0],
                out,
                unit_tokens=1,
            )
            for position, rid in enumerate(request_ids):
                valid = topk[position] >= 0
                got = arenas[layer][out[position][valid].long(), :3]
                units = topk[position][valid]
                expected = torch.stack(
                    (units, torch.full_like(units, layer), torch.full_like(units, rid)), dim=1
                )
                wrong = (got != expected).any(dim=1)
                assert not wrong.any(), (
                    f"request {rid}, history {histories[position]}, layer {layer}: units "
                    f"{units[wrong].tolist()} read {got[wrong].tolist()}"
                )
        manager.update_forward_resources(scheduled, wait_for_input_copy=lambda: None)
        for rid, history in zip(request_ids, histories):
            assert manager.kv_cache_map[rid].history_length == history + 1
    assert permuted == (num_requests > 1)
    for request in active:
        manager.free_resources(request)
    manager.shutdown()


def test_offload_scratch_counts_the_latent_writer(manager):
    with_writer = manager.get_sparse_offload_scratch_bytes(manager.kv_cache_manager_py_config)
    state_only = super(DSACacheManagerV2, manager).get_sparse_offload_scratch_bytes(
        manager.kv_cache_manager_py_config
    )
    assert with_writer - state_only == 2 * 3 * MAX_BATCH_SIZE * 4
