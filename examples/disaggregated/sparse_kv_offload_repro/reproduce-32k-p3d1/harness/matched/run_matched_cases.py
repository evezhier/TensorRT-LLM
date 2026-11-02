# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate full-model sparse residency at matched ready batches without profiling."""

import copy
import json
import os
import re
import subprocess
import time
from pathlib import Path

from analyze_trials import compare
from run_ratios import publish


def run_case(
    root: Path,
    mode: str,
    name: str | None = None,
    agent_bounce_mb: int | None = None,
    quick: bool = False,
) -> None:
    name = name or f"{mode}-matched-p1-d1"
    run = root / "cases" / name
    if (run / "finished.json").exists():
        return
    run.mkdir(parents=True, exist_ok=True)
    (run / "commands").mkdir(exist_ok=True)
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
        "896" if mode == "offload" else "1024",
        "--host-gib",
        "256",
        "--gpu-fraction",
        "0.90" if mode == "offload" else "0.95",
        "--context-max-batch-size",
        "1",
        "--context-instances",
        "1",
        "--output",
        str(run),
        "--capacity-gate",
    ]
    env = os.environ.copy()
    if agent_bounce_mb is not None:
        env["SPARSE_REPRO_AGENT_BOUNCE_MB"] = str(agent_bounce_mb)
        env["SPARSE_REPRO_KV_BOUNCE_MB"] = "0"
    with run.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
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
            batches = (1, 96) if quick else (1, 96, 144, 1204)
            trials += [
                (f"{index + 100:03d}-c{batch}", batch, True) for index, batch in enumerate(batches)
            ]
            if not quick:
                if mode == "offload":
                    trials += [("104-c1696", 1696, True), ("105-c1696-repeat", 1696, True)]
                trials.insert(4, ("200-churn", 128, False))
            for trial, concurrency, gated in trials:
                payload = {
                    "concurrency": concurrency,
                    "requests": 512 if trial == "200-churn" else concurrency,
                    "output_tokens": 160,
                    "gate": gated,
                    "gate_timeout": 900,
                    "plateau_seconds": 120,
                    "profile": False,
                    "purpose": "Fixed ready batches for sparse-cache residency and token parity",
                }
                publish(run / "commands" / f"{trial}.json", payload)
                result = wait(run / f"{trial}-results.json")
                print(
                    json.dumps(
                        {
                            "run": name,
                            "trial": trial,
                            "passed": result["all_requests_passed"],
                            "seconds": result["seconds"],
                            "peak_decode": result["metrics"]["peak_concurrent_decode"],
                        }
                    ),
                    flush=True,
                )
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
        publish(
            run / "finished.json",
            {"mode": mode, "prefill_workers": 1, "generation_workers": 1, "profiling": False},
        )
    print("MATCHED_COMPLETE", name, flush=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    if os.environ.get("SPARSE_REPRO_TRANSPORT_ACCURACY") == "1":
        cases = (
            ("offload-direct-a-matched-p1-d1", 0),
            ("offload-direct-b-matched-p1-d1", 0),
            ("offload-agent-matched-p1-d1", 256),
        )
        for name, agent_bounce_mb in cases:
            run_case(root, "offload", name, agent_bounce_mb, quick=True)
        trials = ("000-cold", "001-warm", "100-c1", "101-c96")
        comparisons = []
        for trial in trials:
            results = [
                json.loads((root / "cases" / name / f"{trial}-results.json").read_text())
                for name, _ in cases
            ]
            comparisons.append(
                {
                    "trial": trial,
                    "direct_repeat": compare(results[0], results[1]),
                    "direct_to_agent": compare(results[0], results[2]),
                }
            )
        configs = [
            json.loads((root / "cases" / name / "config.json").read_text()) for name, _ in cases
        ]
        normalized = []
        for config in configs:
            value = copy.deepcopy(config)
            for key in ("context_config", "generation_config"):
                value[key].pop("perf_metrics_output_dir", None)
                transport = value[key]["cache_transceiver_config"]
                for field in (
                    "kv_cache_bounce_size_mb",
                    "agent_bounce_buffer_enable",
                    "agent_bounce_params",
                ):
                    transport.pop(field, None)
            normalized.append(value)
        configs_match = normalized[0] == normalized[1] == normalized[2]
        direct_disabled = all(
            not config[key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
            for config in configs[:2]
            for key in ("context_config", "generation_config")
        )
        agent_run = root / "cases" / cases[2][0]
        agent_config = configs[2]
        agent_enabled = all(
            agent_config[key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
            for key in ("context_config", "generation_config")
        )
        submits = [
            int(value)
            for value in re.findall(
                r"bounce_submit_count=(\d+)", agent_run.with_suffix(".log").read_text()
            )
        ]
        baseline_stable = all(
            row["direct_repeat"]["identical_token_multisets"] for row in comparisons
        )
        agent_matches = all(
            row["direct_to_agent"]["identical_token_multisets"] for row in comparisons
        )
        if not configs_match or not direct_disabled:
            status = "config_mismatch"
        elif not baseline_stable:
            status = "baseline_unstable"
        elif not agent_matches:
            status = "agent_mismatch"
        elif not agent_enabled or not any(count > 0 for count in submits):
            status = "agent_not_engaged"
        else:
            status = "pass"
        publish(
            root / "transport-accuracy.json",
            {
                "status": status,
                "configs_match_except_transport": configs_match,
                "direct_agent_bounce_disabled": direct_disabled,
                "agent_enabled": agent_enabled,
                "agent_bounce_submit_counts": submits,
                "comparisons": comparisons,
            },
        )
        print("TRANSPORT_ACCURACY", status, flush=True)
        if status != "pass":
            raise RuntimeError(f"Transport accuracy check: {status}")
        return
    for mode in ("resident", "offload"):
        run_case(root, mode)


if __name__ == "__main__":
    main()
