# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare prefill ratios on one fixed decode GPU pair, with original model settings."""

import json
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
    for mode in ("resident", "offload"):
        run = root / "cases" / f"{mode}-p{ratio}-d1"
        if (run / "finished.json").exists():
            continue
        run.mkdir(parents=True, exist_ok=True)
        (run / "commands").mkdir(exist_ok=True)
        batch = 896 if mode == "offload" else 1024
        maximum = 1696 if mode == "offload" else 1204
        command = ["bash", str(root / "launch.sh"), str(root / "run_mode.sh"),
                   "--mode", mode, "--input-tokens", "32768", "--output-tokens", "160",
                   "--max-batch-size", str(batch), "--host-gib", "256",
                   "--gpu-fraction", "0.90" if mode == "offload" else "0.95",
                   "--context-max-batch-size", "128", "--context-instances", str(ratio),
                   "--output", str(run)]
        with run.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            publish(run / "launcher.json", {"pid": process.pid, "command": command})

            def wait(path: Path, timeout: float = 3600) -> dict:
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if path.exists():
                        return json.loads(path.read_text())
                    if process.poll() is not None:
                        if path.exists():
                            return json.loads(path.read_text())
                        raise RuntimeError(f"Server exited {process.returncode}; inspect {log.name}")
                    time.sleep(2)
                raise TimeoutError(str(path))

            try:
                wait(run / "ready.json", 2400)
                print(f"RATIO_READY {mode} prefill={ratio} decode=1", flush=True)
                trials = [("000-cold", 1, 1, 160), ("001-warm", 1, 1, 160),
                          ("002-warm-load", 128, 128, 160),
                          ("100-performance-1", maximum, 4096, 160)]
                for name, concurrency, count, output_tokens in trials:
                    publish(run / "commands" / f"{name}.json", {
                        "concurrency": concurrency, "requests": count, "output_tokens": output_tokens,
                        "gate": False, "profile": False,
                        "purpose": (
                            f"Serving comparison with {ratio} prefill workers "
                            f"and decode GPUs {generation_gpus}"
                        ),
                    })
                    result = wait(run / f"{name}-results.json")
                    print(json.dumps({"mode": mode, "prefill_workers": ratio, "trial": name,
                                      "passed": result["all_requests_passed"], "seconds": result["seconds"],
                                      "output_tokens_per_second": result["output_tokens_per_second"],
                                      "peak_decode": result["metrics"]["peak_concurrent_decode"]}), flush=True)
                    if not result["all_requests_passed"]:
                        raise RuntimeError(f"Request failure in {run.name}/{name}")
            finally:
                publish(run / "commands/999-stop.json", {"stop": True})
            code = process.wait(timeout=300)
            if code:
                raise RuntimeError(f"Server cleanup exited {code}")
            wait(run / "stopped.json", 30)
            publish(run / "finished.json", {"mode": mode, "prefill_workers": ratio,
                                            "generation_workers": 1, "generation_gpus": generation_gpus,
                                            "offered_concurrency": maximum})
        print(f"RATIO_COMPLETE {mode} prefill={ratio}", flush=True)


if __name__ == "__main__":
    main()
