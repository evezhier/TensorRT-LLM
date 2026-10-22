# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate full-model sparse residency at matched ready batches without profiling."""

import json
import subprocess
import time
from pathlib import Path

from run_ratios import publish


def run_case(root: Path, mode: str) -> None:
    name = f"{mode}-matched-p3-d1"
    run = root / "cases" / name
    if (run / "finished.json").exists():
        return
    run.mkdir(parents=True, exist_ok=True)
    (run / "commands").mkdir(exist_ok=True)
    command = [
        "bash", str(root / "launch.sh"), str(root / "run_mode.sh"),
        "--mode", mode, "--input-tokens", "32768", "--output-tokens", "160",
        "--max-batch-size", "896" if mode == "offload" else "1024",
        "--host-gib", "384", "--gpu-fraction", "0.90" if mode == "offload" else "0.95",
        "--context-max-batch-size", "1", "--context-instances", "3",
        "--output", str(run), "--capacity-gate",
    ]
    with run.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        publish(run / "launcher.json", {"pid": process.pid, "command": command})

        def wait(path: Path, timeout: float = 2400) -> dict:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if path.exists():
                    return json.loads(path.read_text())
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited {process.returncode}: {log.name}")
                time.sleep(2)
            raise TimeoutError(str(path))

        try:
            wait(run / "ready.json")
            print("MATCHED_READY", name, flush=True)
            trials = [("000-cold", 1, False), ("001-warm", 1, False)]
            trials += [(f"{index + 100:03d}-c{batch}", batch, True)
                       for index, batch in enumerate((1, 96, 144, 1204))]
            if mode == "offload":
                trials += [("104-c1696", 1696, True), ("105-c1696-repeat", 1696, True)]
            for trial, concurrency, gated in trials:
                payload = {
                    "concurrency": concurrency, "requests": concurrency, "output_tokens": 160,
                    "gate": gated, "gate_timeout": 900, "plateau_seconds": 120,
                    "profile": False,
                    "purpose": "Fixed ready batches for sparse-cache residency and token parity",
                }
                publish(run / "commands" / f"{trial}.json", payload)
                result = wait(run / f"{trial}-results.json")
                print(json.dumps({"run": name, "trial": trial,
                                  "passed": result["all_requests_passed"],
                                  "seconds": result["seconds"],
                                  "peak_decode": result["metrics"]["peak_concurrent_decode"]}),
                      flush=True)
                if not result["all_requests_passed"]:
                    raise RuntimeError(f"Request failure: {name}/{trial}")
                if gated:
                    barriers = result["decode_start_barrier"]
                    if len(barriers) != 2 or not all(
                        barrier["passed"] and barrier["completed"] == concurrency
                        for barrier in barriers
                    ):
                        raise RuntimeError(f"Ready-batch gate failure: {name}/{trial}")
        finally:
            publish(run / "commands/999-stop.json", {"stop": True})
        code = process.wait(timeout=600)
        if code:
            raise RuntimeError(f"Matched case cleanup exited {code}: {name}")
        wait(run / "stopped.json", 30)
        publish(run / "finished.json", {"mode": mode, "prefill_workers": 3,
                                        "generation_workers": 1, "profiling": False})
    print("MATCHED_COMPLETE", name, flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    for mode in ("resident", "offload"):
        run_case(root, mode)


if __name__ == "__main__":
    main()
