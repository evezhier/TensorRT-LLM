# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure prefill completion cadence from the headline serving request records."""

import json
from collections import defaultdict
from pathlib import Path

from analyze_decode import distribution

ROOT = Path(__file__).resolve().parent


def main() -> None:
    preflight = json.loads((ROOT / "preflight.json").read_text())
    ratio = preflight.get("prefill_workers", len(preflight["gpus"]) // 2 - 1)
    modes = {}
    for mode in ("resident", "offload"):
        source = ROOT / "cases" / f"{mode}-p{ratio}-d1" / "100-performance-1-perf.json"
        records = json.loads(source.read_text())
        ids = {
            r["ctx_request_id"]
            for r in records
            if "perf_metrics-generation-" in r["source_file"]
            and len(r.get("time_breakdown_metrics", {}).get("step_metrics", [])) == 159
        }
        contexts = [
            r
            for r in records
            if "perf_metrics-context-" in r["source_file"] and r["ctx_request_id"] in ids
        ]
        assert len(contexts) == len(ids) == 4096
        by_worker = defaultdict(list)
        for row in contexts:
            by_worker[row["source_file"]].append(row)
        assert len(by_worker) == ratio
        ready = [r["perf_metrics"]["timing_metrics"]["kv_cache_transfer_start"] for r in contexts]
        start, end = min(ready), max(ready)
        middle_start = start + 0.1 * (end - start)
        middle_end = start + 0.9 * (end - start)
        workers = []
        for name, members in sorted(by_worker.items()):
            times = sorted(
                r["perf_metrics"]["timing_metrics"]["kv_cache_transfer_start"] for r in members
            )
            intervals = [
                1000 * (b - a)
                for a, b in zip(times, times[1:])
                if a >= middle_start and b <= middle_end
            ]
            count = sum(middle_start <= t <= middle_end for t in times)
            workers.append(
                {
                    "source_file": name,
                    "requests": len(members),
                    "central_window_ready_requests": count,
                    "central_window_ready_requests_per_second": count / (middle_end - middle_start),
                    "central_window_inter_ready_ms": distribution(intervals),
                    "gpu_forward_ms": distribution(
                        [r["time_breakdown_metrics"]["ctx_gpu_forward_time"] for r in members]
                    ),
                }
            )
        modes[mode] = {
            "requests": len(contexts),
            "workers": workers,
            "central_window_seconds": middle_end - middle_start,
            "central_window_ready_requests_per_second": sum(
                r["central_window_ready_requests_per_second"] for r in workers
            ),
            "first_to_last_kv_ready_seconds": end - start,
        }
    output = {
        "scope": (
            f"Unprofiled 32K/160 {ratio}:1 runs after the residency fix. "
            "Context prefix reuse is enabled with repeated synthetic prompts; "
            "these are not fresh unique 32K-prefill rates."
        ),
        "definition": (
            "KV-ready means the context request's kv_cache_transfer_start "
            "timestamp, before transfer. Central window is 10% through 90% "
            "of the first-to-last KV-ready interval; all prefill workers use "
            "the same window."
        ),
        "modes": modes,
    }
    (ROOT / "prefill-feed.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({m: r["central_window_ready_requests_per_second"] for m, r in modes.items()}))


if __name__ == "__main__":
    main()
