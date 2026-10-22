# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save allocated KV pools and sparse cache geometry once after initialization."""

import dataclasses
import functools
import json
import os
from pathlib import Path


def install(module) -> None:
    """Observe allocation metadata without instrumenting decode iterations."""
    output = Path(os.environ["SWEEP_METRICS_DIR"])
    if module.__name__.endswith(".sparse_offload_memory"):
        original = module.solve_sparse_offload_capacity

        @functools.wraps(original)
        def solve(*args, **kwargs):
            capacity = original(*args, **kwargs)
            data = dataclasses.asdict(capacity)
            data.update({"fits": capacity.fits, "limited_by": capacity.limited_by})
            (output / f"capacity-estimate-pid{os.getpid()}.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )
            return capacity

        module.solve_sparse_offload_capacity = solve
        return
    original = module.DeepseekV4CacheManager.__init__

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.is_estimating_kv_cache:
            return
        from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
            build_page_table_from_manager,
        )
        from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import GPU_LEVEL
        from tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory import (
            sparse_arena_bytes,
            sparse_arena_shapes,
            sparse_history_bytes_per_block,
        )

        config = self.kv_cache_manager_py_config
        table = build_page_table_from_manager(self)
        input_tokens = int(os.environ["SWEEP_INPUT_TOKENS"])
        record = {
            "rank": self.mapping.rank,
            "pid": os.getpid(),
            "input_tokens": input_tokens,
            "tokens_per_block": self.tokens_per_block,
            "max_batch_size": self.max_batch_size,
            "max_admissible_sequences": self.max_admissible_sequences,
            "gpu_pool_quota_bytes": self.impl.get_quota(GPU_LEVEL),
            "sparse_arena_bytes": sparse_arena_bytes(config),
            "sparse_batch_buffer_bytes": self._sparse_batch_buffer_bytes,
            "sparse_batch_staging_bytes": self._sparse_batch_staging_bytes,
            "host_history_budget_bytes": self._host_history_budget,
            "host_history_pool_bytes": self._get_host_backed_pool_bytes(),
            "host_history_capacity_blocks": self._host_history_capacity_blocks,
            "host_history_bytes_per_block": sparse_history_bytes_per_block(config.layers),
            "gpu_pool_payload_bytes_per_request": self.get_request_device_bytes(input_tokens + 160),
            "sparse_arena_shapes": [
                dataclasses.asdict(shape)
                for shape in sparse_arena_shapes(
                    config.layers, tokens_per_block=self.tokens_per_block
                )
            ],
            "physical_pools": [
                {
                    "group": index,
                    "memory_type": pool.memory_type,
                    "slot_bytes": pool.slot_bytes,
                    "num_slots": pool.num_slots,
                    "payload_bytes": pool.slot_bytes * pool.num_slots,
                }
                for index, group in enumerate(table.pool_groups)
                for pool in group.pools
            ],
        }
        (output / f"memory-allocation-rank{self.mapping.rank}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )

    module.DeepseekV4CacheManager.__init__ = initialize

    original_shutdown = module.DeepseekV4CacheManager.shutdown

    @functools.wraps(original_shutdown)
    def shutdown(self, *args, **kwargs):
        batch = self._sparse_batch
        if batch is not None and not self.is_estimating_kv_cache:
            import torch
            torch.cuda.synchronize(self._stream.device)
            hits, misses = batch.peek_hit_miss_counter()
            (output / f"sparse-counters-rank{self.mapping.rank}.json").write_text(
                json.dumps({"pid": os.getpid(), "hits": hits, "misses": misses,
                            "hit_ratio": hits / (hits + misses) if hits + misses else None,
                            "sparse_arena_bytes": batch.sparse_arena_bytes,
                            "scope": "Entire process including warmups; sampled after all trials"},
                           indent=2) + "\n"
            )
        return original_shutdown(self, *args, **kwargs)

    module.DeepseekV4CacheManager.shutdown = shutdown
