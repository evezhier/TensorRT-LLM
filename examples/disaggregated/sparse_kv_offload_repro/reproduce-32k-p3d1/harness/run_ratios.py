# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare prefill ratios on one fixed decode GPU pair, with original model settings."""

import json
import os
import subprocess
import time
from pathlib import Path


def publish(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parent
    preflight = json.loads((root / "preflight.json").read_text())
    ratio = preflight.get("prefill_workers", len(preflight["gpus"]) // 2 - 1)
    generation_gpus = list(range(2 * ratio, 2 * ratio + 2))
    cases = (
        (
            ("resident", "resident", None),
            ("offload-direct", "offload", 0),
            ("offload-agent", "offload", 256),
        )
        if preflight["variant"] == "compare"
        else (("resident", "resident", None), ("offload", "offload", None))
    )
    compare_variant = preflight["variant"] == "compare"
    perf_concurrency = os.environ.get("SPARSE_REPRO_PERF_CONCURRENCY")
    perf_requests = int(os.environ.get("SPARSE_REPRO_PERF_REQUESTS", "4096"))
    if perf_requests <= 0:
        raise ValueError("SPARSE_REPRO_PERF_REQUESTS must be positive")
    for name, mode, agent_bounce_mb in cases:
        run = root / "cases" / f"{name}-p{ratio}-d1"
        if (run / "finished.json").exists():
            continue
        run.mkdir(parents=True, exist_ok=True)
        (run / "commands").mkdir(exist_ok=True)
        batch = 896 if mode == "offload" or compare_variant else 1024
        maximum = 1024 if compare_variant else (1696 if mode == "offload" else 1204)
        if perf_concurrency is not None:
            maximum = int(perf_concurrency)
        if maximum <= 0 or maximum > perf_requests:
            raise ValueError(
                "SPARSE_REPRO_PERF_CONCURRENCY must be between 1 and SPARSE_REPRO_PERF_REQUESTS"
            )
        command = [
            "bash",
            str(root / "launch.sh"),
            str(root / "run_mode.sh"),
            "--mode",
            mode,
            "--input-tokens",
            "32768",
            "--output-tokens",
            "160",
            "--max-batch-size",
            str(batch),
            "--host-gib",
            "256",
            "--gpu-fraction",
            "0.90" if mode == "offload" or compare_variant else "0.95",
            "--context-max-batch-size",
            "128",
            "--context-instances",
            str(ratio),
            "--output",
            str(run),
        ]
        env = os.environ.copy()
        if agent_bounce_mb is not None:
            env["SPARSE_REPRO_AGENT_BOUNCE_MB"] = str(agent_bounce_mb)
            env["SPARSE_REPRO_KV_BOUNCE_MB"] = "0"
        with run.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            publish(run / "launcher.json", {"pid": process.pid, "command": command})

            def wait(path: Path, timeout: float = 3600) -> dict:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if path.exists():
                        return json.loads(path.read_text())
                    if process.poll() is not None:
                        if path.exists():
                            return json.loads(path.read_text())
                        raise RuntimeError(
                            f"Server exited {process.returncode}; inspect {log.name}"
                        )
                    time.sleep(2)
                raise TimeoutError(str(path))

            try:
                wait(run / "ready.json", 2400)
                print(f"RATIO_READY {name} prefill={ratio} decode=1", flush=True)
                trials = [
                    ("000-cold", 1, 1, 160),
                    ("001-warm", 1, 1, 160),
                    ("002-warm-load", 128, 128, 160),
                    ("100-performance-1", maximum, perf_requests, 160),
                ]
                for name, concurrency, count, output_tokens in trials:
                    publish(
                        run / "commands" / f"{name}.json",
                        {
                            "concurrency": concurrency,
                            "requests": count,
                            "output_tokens": output_tokens,
                            "gate": False,
                            "profile": False,
                            "purpose": (
                                f"Serving comparison with {ratio} prefill workers "
                                f"and decode GPUs {generation_gpus}"
                            ),
                        },
                    )
                    result = wait(run / f"{name}-results.json")
                    print(
                        json.dumps(
                            {
                                "mode": mode,
                                "prefill_workers": ratio,
                                "trial": name,
                                "passed": result["all_requests_passed"],
                                "seconds": result["seconds"],
                                "output_tokens_per_second": result["output_tokens_per_second"],
                                "peak_decode": result["metrics"]["peak_concurrent_decode"],
                            }
                        ),
                        flush=True,
                    )
                    if not result["all_requests_passed"]:
                        raise RuntimeError(f"Request failure in {run.name}/{name}")
            finally:
                publish(run / "commands/999-stop.json", {"stop": True})
            code = process.wait(timeout=300)
            if code:
                raise RuntimeError(f"Server cleanup exited {code}")
            wait(run / "stopped.json", 30)
            publish(
                run / "finished.json",
                {
                    "mode": mode,
                    "prefill_workers": ratio,
                    "generation_workers": 1,
                    "generation_gpus": generation_gpus,
                    "offered_concurrency": maximum,
                },
            )
        print(f"RATIO_COMPLETE {name} prefill={ratio}", flush=True)


if __name__ == "__main__":
    main()
