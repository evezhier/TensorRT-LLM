# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the built runtime and record reproducible inputs before starting workers."""

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main() -> None:
    import torch
    from defs.accuracy.test_disaggregated_serving import launch_disaggregated_llm

    import tensorrt_llm
    from tensorrt_llm.runtime.kv_cache_manager_v2 import Batch

    repo = Path(os.environ["E2E_REPO"]).resolve()
    output = Path(os.environ["REPRO_OUTPUT"])
    variant = sys.argv[1]
    expected_gpus = 8 if variant == "serving" else 4
    if torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(f"{variant} requires exactly {expected_gpus} visible GPUs")
    if Path(tensorrt_llm.__file__).resolve().parent != repo / "tensorrt_llm":
        raise RuntimeError("TensorRT-LLM was imported from a different checkout")
    if not hasattr(Batch, "reorder"):
        raise RuntimeError("Rebuild the native bindings: Batch.reorder is missing")
    if not callable(launch_disaggregated_llm):
        raise RuntimeError("Disaggregated integration-test launcher is unavailable")
    config_path = Path(os.environ["LLM_MODELS_ROOT"]) / "DeepSeek-V4-Flash/config.json"
    config = json.loads(config_path.read_text())
    if config["num_hidden_layers"] != 43 or config["max_position_embeddings"] < 33024:
        raise RuntimeError("Expected the full 43-layer DeepSeek V4 Flash checkpoint")
    packages = {}
    for name in ("torch", "tensorrt_llm", "transformers", "nixl", "aiohttp", "openai"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    info = {
        "variant": variant,
        "repo": str(repo),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "working_tree": subprocess.check_output(
            ["git", "status", "--short"], cwd=repo, text=True
        ),
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(expected_gpus)],
        "batch_reorder_available": True,
    }
    (output / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "source-working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    )
    print("Runtime and model preflight passed")


if __name__ == "__main__":
    main()
