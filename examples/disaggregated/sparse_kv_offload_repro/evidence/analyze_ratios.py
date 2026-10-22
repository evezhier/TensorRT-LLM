# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze completed ratio trials without reading or draining live worker metrics."""

import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from analyze_decode import analyze, distribution
from run_ratios import publish

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from analyze_trials import compare, native_transfer_summary  # noqa: E402


def gpu_summary(run: Path, summary: dict) -> dict:
    ready = json.loads((run / "ready.json").read_text())
    zone = datetime.fromisoformat(ready["local_clock"]).tzinfo
    first = datetime.fromisoformat(summary["started_at_utc"])
    last = datetime.fromisoformat(summary["completed_at_utc"])
    devices = defaultdict(list)
    with (ROOT / "gpu-telemetry.csv").open() as stream:
        for record in csv.DictReader(stream, skipinitialspace=True):
            if not record.get("power.draw [W]"):
                continue
            when = datetime.strptime(record["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=zone)
            if first <= when <= last:
                devices[int(record["index"])].append(record)
    return {str(device): {
        "uuid": records[0]["uuid"], "sample_count": len(records),
        "utilization_percent": distribution([float(r["utilization.gpu [%]"].split()[0]) for r in records]),
        "power_watts": distribution([float(r["power.draw [W]"].split()[0]) for r in records]),
        "peak_memory_mib": max(float(r["memory.used [MiB]"].split()[0]) for r in records),
    } for device, records in sorted(devices.items())}


def main() -> None:
    rows = []
    for mode in ("resident", "offload"):
        variants = [("default", 1), ("default", 3)]
        if mode == "offload":
            variants.append(("prefillcap", 1))
        for variant, ratio in variants:
            name = f"{mode}-p{ratio}-d1" if variant == "default" else "offload-prefillcap-p1-d1"
            run = ROOT / "cases" / name
            for trial in ("001-warm", "100-performance-1"):
                source = run / f"{trial}-results.json"
                if not source.exists():
                    continue
                target = run / f"{trial}-decode-analysis.json"
                if target.exists() and json.loads(target.read_text()).get("analysis_version") == 4:
                    summary = json.loads(target.read_text())
                else:
                    summary = analyze(source)
                    result = json.loads(source.read_text())
                    assert result["all_requests_passed"]
                    assert result["completed_requests"] == result["command"]["requests"]
                    assert all(len(r["token_ids"]) == result["output_tokens"] for r in result["requests"])
                    native = native_transfer_summary(run, result)
                    native.pop("_selected_request_ids", None)
                    summary.update({
                        "mode": mode, "prefill_workers": ratio, "generation_workers": 1,
                        "generation_gpus": [6, 7], "trial": trial, "native_transfer": native,
                        "started_at_utc": result["started_at_utc"],
                        "completed_at_utc": result["completed_at_utc"],
                        "client_seconds": result["seconds"],
                        "latency_seconds": result["latency_seconds"],
                        "ttft_seconds": result["ttft_seconds"],
                        "tpot_seconds": result["tpot_seconds"],
                        "metrics": result["metrics"],
                    })
                    publish(target, summary)
                if "context_worker_request_counts" not in summary:
                    perf = json.loads(source.with_name(source.name.replace("-results", "-perf")).read_text())
                    generation_ids = {
                        p.get("ctx_request_id") for p in perf
                        if "perf_metrics-generation-" in p["source_file"]
                        and len(p.get("time_breakdown_metrics", {}).get("step_metrics", [])) == 159
                    }
                    context = [p for p in perf if "perf_metrics-context-" in p["source_file"]
                               and p.get("ctx_request_id") in generation_ids]
                    context_events = {}
                    for request in context:
                        for step in request.get("time_breakdown_metrics", {}).get("ctx_chunk_metrics", []):
                            key = (request["source_file"], step["forward_start_time"], step["forward_end_time"])
                            context_events[key] = step["gpu_forward_time"]
                    summary["context_gpu_forward_ms"] = distribution(list(context_events.values()))
                    summary["context_worker_request_counts"] = dict(Counter(p["source_file"] for p in context))
                    publish(target, summary)
                summary["gpu_telemetry"] = gpu_summary(run, summary)
                summary["variant"] = variant
                summary["run_name"] = name
                config = json.loads((run / "config.json").read_text())
                assert config["physical_generation_gpu_ids"] == [6, 7]
                assert config["physical_context_gpu_ids"] == list(range(2 * ratio))
                assert config["disaggregation_config"]["context_servers"]["num_instances"] == ratio
                summary["config"] = config
                publish(target, summary)
                compact = {k: v for k, v in summary.items() if k not in ("time_series", "partial_perf_active_window")}
                compact["generation_active_intervals"] = {
                    k: v for k, v in compact["generation_active_intervals"].items()
                    if k != "time_at_active_count_seconds"
                }
                rows.append(compact)
    parity = []
    for trial in ("000-cold", "001-warm", "100-performance-1"):
        for left_name, right_name in (
            ("resident-p1-d1", "offload-p1-d1"),
            ("resident-p3-d1", "offload-p3-d1"),
            ("resident-p1-d1", "resident-p3-d1"),
            ("offload-p1-d1", "offload-p3-d1"),
            ("offload-p1-d1", "offload-prefillcap-p1-d1"),
        ):
            left = ROOT / "cases" / left_name / f"{trial}-results.json"
            right = ROOT / "cases" / right_name / f"{trial}-results.json"
            if left.exists() and right.exists():
                parity.append({"trial": trial, "left": left_name, "right": right_name,
                               **compare(json.loads(left.read_text()), json.loads(right.read_text()))})
    publish(ROOT / "ratio-analysis.json", {"trials": rows, "parity": parity})
    flat = []
    for row in rows:
        item = {k: row[k] for k in ("mode", "prefill_workers", "variant", "trial", "requests",
                                    "decode_worker_tokens_per_second", "e2e_output_tokens_per_second",
                                    "peak_decode_requests")}
        item.update({
            "arrival_requests_per_second": row.get("arrival_requests_per_second"),
            "mean_active_requests": row["generation_active_intervals"].get("mean_active_requests"),
            "mean_decode_seconds": row["generation_request_compute_seconds"]["mean"],
            "p50_decode_seconds": row["generation_request_compute_seconds"]["p50"],
            "p50_gpu_forward_ms": row["generation_forward_event_ms"]["p50"],
            "p50_e2e_seconds": row["latency_seconds"]["p50"],
            "p50_ttft_seconds": row["ttft_seconds"]["p50"],
            "p50_send_transfer_ms": row["native_transfer"]["send_transfer_phase_ms"]["p50"],
        })
        flat.append(item)
        print(json.dumps(item), flush=True)
    if flat:
        with (ROOT / "ratio-metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)


if __name__ == "__main__":
    main()
