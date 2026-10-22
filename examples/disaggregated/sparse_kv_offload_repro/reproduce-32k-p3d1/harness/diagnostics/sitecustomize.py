# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install startup allocation measurements and opt-in diagnostic synchronization."""

import importlib.abc
import importlib.machinery
import json
import os
import sys
from pathlib import Path

if os.environ.get("SWEEP_METRICS_DIR") and os.environ.get("TRTLLM_KVCACHE_TIME_OUTPUT_PATH"):
    (Path(os.environ["SWEEP_METRICS_DIR"]) / f"transfer-source-pid{os.getpid()}.json").write_text(
        json.dumps({"path": os.environ["TRTLLM_KVCACHE_TIME_OUTPUT_PATH"]}) + "\n"
    )


class GateLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        if module.__name__.endswith(".openai_client"):
            from http_connection_policy import install
        elif module.__name__.endswith((".sparse_offload_memory", ".deepseek_v4.cache_manager")):
            from memory_metrics import install
        else:
            from parity_gate import install
        install(module)


class GateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        targets = set()
        if os.environ.get("SWEEP_HTTP_FRESH_CONNECTIONS") == "1":
            targets.add("tensorrt_llm.serve.openai_client")
        if os.environ.get("LONG_PARITY_GATE_PATH"):
            targets.update(
                (
                    "tensorrt_llm._torch.disaggregation.transceiver",
                    "tensorrt_llm._torch.disaggregation.orchestration.coordinator",
                )
            )
        if os.environ.get("SWEEP_METRICS_DIR"):
            targets.update(
                (
                    "tensorrt_llm._torch.pyexecutor.kv_cache.sparse_offload_memory",
                    "tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.cache_manager",
                )
            )
        if fullname not in targets:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = GateLoader(spec.loader)
        return spec


if (os.environ.get("LONG_PARITY_GATE_PATH") or os.environ.get("SWEEP_METRICS_DIR")
        or os.environ.get("SWEEP_HTTP_FRESH_CONNECTIONS") == "1"):
    sys.meta_path.insert(0, GateFinder())
