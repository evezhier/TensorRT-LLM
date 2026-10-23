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


def git_provenance(repo: Path, output: Path) -> tuple[str | None, str | None]:
    try:
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repo, text=True, capture_output=True, check=False
        )
    except FileNotFoundError:
        return None, None
    if git_root.returncode != 0 or Path(git_root.stdout.strip()).resolve() != repo:
        return None, None

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    working_tree = subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True)
    (output / "source-working-tree.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    )
    return head, working_tree


def main() -> None:
    import torch
    from defs.accuracy.test_disaggregated_serving import launch_disaggregated_llm

    import tensorrt_llm
    from tensorrt_llm.runtime.kv_cache_manager_v2 import Batch

    repo = Path(os.environ["REPO"]).resolve()
    output = Path(os.environ["OUTPUT"])
    variant = sys.argv[1]
    expected_gpus = 8 if variant == "serving" else 4
    if torch.cuda.device_count() != expected_gpus:
        raise RuntimeError(f"{variant} requires exactly {expected_gpus} visible GPUs")
    package_dir = Path(tensorrt_llm.__file__).resolve().parent
    wheel_path = os.environ.get("WHEEL")
    if wheel_path:
        if package_dir == repo / "tensorrt_llm":
            raise RuntimeError("Wheel mode imported TensorRT-LLM from the source checkout")
    elif package_dir != repo / "tensorrt_llm":
        raise RuntimeError("TensorRT-LLM was imported from a different checkout")
    if not hasattr(Batch, "reorder"):
        raise RuntimeError("Rebuild the native bindings: Batch.reorder is missing")
    if not callable(launch_disaggregated_llm):
        raise RuntimeError("Disaggregated integration-test launcher is unavailable")
    wheel_sha256 = None
    if wheel_path:
        with Path(wheel_path).open("rb") as wheel_file:
            wheel_sha256 = hashlib.file_digest(wheel_file, "sha256").hexdigest()
    config_path = Path(os.environ["MODEL_DIR"]) / "config.json"
    config = json.loads(config_path.read_text())
    if config["num_hidden_layers"] != 43 or config["max_position_embeddings"] < 33024:
        raise RuntimeError("Expected the full 43-layer DeepSeek V4 Flash checkpoint")
    packages = {}
    for name in ("torch", "tensorrt_llm", "transformers", "nixl", "aiohttp", "openai"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    head, working_tree = git_provenance(repo, output)
    info = {
        "variant": variant,
        "repo": str(repo),
        "package_dir": str(package_dir),
        "wheel_path": wheel_path,
        "wheel_sha256": wheel_sha256,
        "model_dir": str(config_path.parent.resolve()),
        "head": head,
        "working_tree": working_tree,
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "python": sys.version,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(index) for index in range(expected_gpus)],
        "batch_reorder_available": True,
    }
    (output / "preflight.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Runtime and model preflight passed")


if __name__ == "__main__":
    main()
