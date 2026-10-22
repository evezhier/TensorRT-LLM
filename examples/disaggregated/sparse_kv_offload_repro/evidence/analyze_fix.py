# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize completed residency-fix trials without querying running workers."""

import json
from datetime import datetime, timezone
from pathlib import Path

from analyze_decode import analyze
from analyze_trials import compare, native_transfer_summary
from run_ratios import publish

ROOT = Path(__file__).resolve().parent


def summarize(path: Path) -> dict:
    target = path.with_name(path.name.replace("-results", "-decode-analysis"))
    if target.exists():
        summary = json.loads(target.read_text())
    else:
        summary = analyze(path)
        result = json.loads(path.read_text())
        native = native_transfer_summary(path.parent, result)
        native.pop("_selected_request_ids", None)
        summary.update({
            "native_transfer": native,
            "client_seconds": result["seconds"],
            "latency_seconds": result["latency_seconds"],
            "ttft_seconds": result["ttft_seconds"],
            "tpot_seconds": result["tpot_seconds"],
            "metrics": result["metrics"],
            "decode_start_barrier": result.get("decode_start_barrier"),
            "started_at_utc": result["started_at_utc"],
            "completed_at_utc": result["completed_at_utc"],
        })
        publish(target, summary)
    compact = {k: v for k, v in summary.items()
               if k not in ("time_series", "partial_perf_active_window")}
    compact["generation_active_intervals"] = {
        k: v for k, v in compact["generation_active_intervals"].items()
        if k != "time_at_active_count_seconds"
    }
    return compact


def main() -> None:
    runs, parity = [], []
    for cases in (ROOT / "cases", ROOT / "fallback-p1/cases"):
        if not cases.exists():
            continue
        for run in sorted(cases.iterdir()):
            if not run.is_dir() or not (run / "ready.json").exists():
                continue
            config = json.loads((run / "config.json").read_text())
            trials = []
            for result in sorted(run.glob("*-results.json")):
                if not result.with_name(result.name.replace("-results", "-metrics")).exists():
                    continue
                summary = summarize(result)
                summary["trial"] = result.name.removesuffix("-results.json")
                trials.append(summary)
                if run.name.startswith("offload"):
                    baseline = run.with_name(run.name.replace("offload", "resident", 1)) / result.name
                    if baseline.exists():
                        parity.append({
                            "left": str(baseline), "right": str(result),
                            **compare(json.loads(baseline.read_text()), json.loads(result.read_text())),
                        })
                if result.name == "105-c1696-repeat-results.json":
                    baseline = run / "104-c1696-results.json"
                    parity.append({
                        "left": str(baseline), "right": str(result),
                        **compare(json.loads(baseline.read_text()), json.loads(result.read_text())),
                    })
                print(json.dumps({
                    "run": run.name, "trial": summary["trial"],
                    "decode_tps": summary["decode_worker_tokens_per_second"],
                    "gpu_p50_ms": summary["generation_forward_event_ms"].get("p50"),
                    "mean_active": summary["generation_active_intervals"].get("mean_active_requests"),
                    "peak": summary["peak_decode_requests"],
                }), flush=True)
            counters = [{"file": str(p), **json.loads(p.read_text())}
                        for p in sorted(run.glob("sparse-counters-rank*.json"))]
            hits, misses = (sum(c[key] for c in counters) for key in ("hits", "misses"))
            runs.append({
                "directory": str(run), "finished": (run / "finished.json").exists(),
                "config": config, "trials": trials, "sparse_counters": counters,
                "total_sparse_hits": hits, "total_sparse_misses": misses,
                "sparse_hit_rate_percent": 100 * hits / (hits + misses) if hits + misses else None,
            })
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs, "parity": parity,
        "counter_scope": "Whole worker process including warmups; collected at shutdown",
        "decode_throughput_definition": (
            "159 generation tokens per completed request divided by first decode forward start "
            "to final decode token when complete request records are available. Otherwise useful "
            "scheduled generation tokens between first and last iteration starts, excluding the "
            "last iteration. Prefill produces the first output token."
        ),
    }
    publish(ROOT / "fix-analysis.json", report)
    print(json.dumps({"parity": parity, "hit_rates": [
        {"run": Path(run["directory"]).name, "percent": run["sparse_hit_rate_percent"]}
        for run in runs if run["sparse_counters"]
    ]}), flush=True)


if __name__ == "__main__":
    main()
