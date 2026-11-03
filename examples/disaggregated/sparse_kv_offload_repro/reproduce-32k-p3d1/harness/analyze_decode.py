# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derive useful decoder throughput and arrival pressure from saved worker records."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        pos = fraction * (len(ordered) - 1)
        low = int(pos)
        return ordered[low] + (ordered[min(low + 1, len(ordered) - 1)] - ordered[low]) * (pos - low)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def active_window(intervals: list[tuple[float, float]]) -> dict:
    events = defaultdict(int)
    for start, end in intervals:
        assert end >= start
        events[start] += 1
        events[end] -= 1
    if not events:
        return {}
    points = sorted(events)
    active, area, occupied, peak = 0, 0.0, 0.0, 0
    histogram = defaultdict(float)
    for index, point in enumerate(points[:-1]):
        active += events[point]
        duration = points[index + 1] - point
        area += active * duration
        occupied += duration if active else 0
        histogram[active] += duration
        peak = max(active, peak)
    window = points[-1] - points[0]
    return {
        "start": points[0],
        "end": points[-1],
        "seconds": window,
        "mean_active_requests": area / window,
        "peak_active_requests": peak,
        "empty_seconds": window - occupied,
        "active_request_seconds": area,
        "time_at_active_count_seconds": dict(sorted(histogram.items())),
    }


def analyze(path: Path) -> dict:
    result = json.loads(path.read_text())
    raw = json.loads(path.with_name(path.name.replace("-results", "-metrics")).read_text())
    grouped = defaultdict(dict)
    for row in raw["generation_servers"]:
        rank = row.get("attentionDpRank", row.get("rank", 0))
        grouped[row["iter"]][rank] = row
    steps = []
    for iteration, ranks in sorted(grouped.items()):
        count = sum(r["inflightBatchingStats"]["numGenRequests"] for r in ranks.values())
        if not count:
            continue
        row = ranks.get(0, next(iter(ranks.values())))
        stamp = datetime.strptime(row["timestamp"], "%m-%d-%Y %H:%M:%S.%f").timestamp()
        steps.append((iteration, stamp, count, row))
    assert steps and all(steps[i + 1][1] > steps[i][1] for i in range(len(steps) - 1))
    expected = result["completed_requests"] * (result["output_tokens"] - 1)
    scheduled = sum(s[2] for s in steps)
    assert scheduled <= expected, (str(path), scheduled, expected)
    window = steps[-1][1] - steps[0][1]
    measured_tokens = sum(s[2] for s in steps[:-1])
    cadence = [(b[1] - a[1]) * 1000 for a, b in zip(steps, steps[1:])]
    perf = json.loads(path.with_name(path.name.replace("-results", "-perf")).read_text())
    generation = [r for r in perf if "perf_metrics-generation-" in r["source_file"]]
    context = [r for r in perf if "perf_metrics-context-" in r["source_file"]]
    generation = [
        r
        for r in generation
        if len(r.get("time_breakdown_metrics", {}).get("step_metrics", []))
        == result["output_tokens"] - 1
    ]
    ctx_by_id = {r.get("ctx_request_id"): r for r in context}
    durations, token_gaps, intervals, arrivals = [], [], [], []
    stage_times = defaultdict(list)
    forward_events = {}
    time_series = []
    for record in generation:
        timing = record["perf_metrics"]["timing_metrics"]
        request_steps = record["time_breakdown_metrics"]["step_metrics"]
        assert [s["iter"] for s in request_steps] == list(range(2, result["output_tokens"] + 1))
        start = request_steps[0]["forward_start_time"]
        end = request_steps[-1]["token_time"]
        intervals.append((start, end))
        durations.append(end - start)
        arrivals.append(start)
        gaps = [b["token_time"] - a["token_time"] for a, b in zip(request_steps, request_steps[1:])]
        token_gaps.append(statistics.mean(gaps) * 1000)
        for step in request_steps:
            forward_events[(step["forward_start_time"], step["forward_end_time"])] = step[
                "gpu_forward_time"
            ]
        for name, first, last in (
            ("generation_receive_lifecycle", "kv_cache_transfer_start", "kv_cache_transfer_end"),
            (
                "receive_complete_to_first_scheduled",
                "kv_cache_transfer_end",
                "first_scheduled_time",
            ),
            ("generation_first_scheduled_to_last_token", "first_scheduled_time", "last_token_time"),
        ):
            if first in timing and last in timing:
                stage_times[name].append(timing[last] - timing[first])
        ctx = ctx_by_id.get(record.get("ctx_request_id"))
        ct = ctx["perf_metrics"]["timing_metrics"] if ctx else {}
        for name, value in (
            ("context_queue", ct.get("first_scheduled_time", 0) - ct.get("arrival_time", 0)),
            (
                "context_scheduled_to_first_token",
                ct.get("first_token_time", 0) - ct.get("first_scheduled_time", 0),
            ),
            (
                "context_ready_to_receive_complete",
                timing.get("kv_cache_transfer_end", 0) - ct.get("kv_cache_transfer_start", 0),
            ),
        ):
            if ct and value >= 0:
                stage_times[name].append(value)
        time_series.append(
            {
                "start": start,
                "end": end,
                "receive_end": timing.get("kv_cache_transfer_end"),
                "context_scheduled": ct.get("first_scheduled_time"),
                "context_ready": ct.get("kv_cache_transfer_start"),
            }
        )
    active = active_window(intervals)
    covered = len(generation) == result["completed_requests"]
    context_steps = {}
    for row in raw["context_servers"]:
        batch = row["inflightBatchingStats"]
        if batch["numContextRequests"]:
            context_steps[(row.get("worker_port"), row["iter"], row["timestamp"])] = row
    context_counts = Counter(
        r["inflightBatchingStats"]["numContextRequests"] for r in context_steps.values()
    )
    summary = {
        "analysis_version": 4,
        "result_file": str(path),
        "input_tokens": result["input_tokens"],
        "mode": result["mode"],
        "command": result["command"],
        "requests": result["completed_requests"],
        "output_tokens": result["output_tokens"],
        "all_requests_passed": result["all_requests_passed"],
        "e2e_output_tokens_per_second": result["output_tokens_per_second"],
        "generation_tokens_per_client_wall_second": expected / result["seconds"],
        "decode_worker_tokens_per_second": measured_tokens / window,
        "scheduled_tokens_per_iteration_window_second": measured_tokens / window,
        "scheduled_tokens_in_iteration_window": measured_tokens,
        "iteration_start_window_seconds": window,
        "iteration_token_coverage_complete": scheduled == expected,
        "decode_worker_tokens_in_window": measured_tokens,
        "decode_worker_window_seconds": window,
        "decode_iterations": len(steps),
        "scheduled_generation_tokens": scheduled,
        "expected_generation_tokens": expected,
        "decode_step_cadence_ms": distribution(cadence),
        "iteration_lifecycle_ms": distribution([s[3]["iterLatencyMS"] for s in steps]),
        "host_step_ms": distribution(
            [s[3]["hostStepTimeMS"] for s in steps if "hostStepTimeMS" in s[3]]
        ),
        "scheduled_decode_count": distribution([s[2] for s in steps]),
        "peak_decode_requests": max(s[2] for s in steps),
        "request_perf_coverage": {
            "count": len(generation),
            "expected": result["completed_requests"],
            "complete": covered,
        },
        "generation_request_compute_seconds": distribution(durations),
        "generation_request_mean_token_interval_ms": distribution(token_gaps),
        "generation_forward_event_ms": distribution(list(forward_events.values())),
        "generation_active_intervals": active if covered else {"complete": False},
        "partial_perf_active_window": active if not covered else None,
        "stage_seconds": {k: distribution(v) for k, v in stage_times.items()},
        "context_batch_histogram": dict(sorted(context_counts.items())),
        "context_peak_queue": max(
            (r.get("numQueuedRequests", 0) for r in raw["context_servers"]), default=0
        ),
        "time_series": time_series if covered else [],
    }
    if covered:
        summary["generation_tokens_per_forward_to_last_token_window_second"] = (
            expected / active["seconds"]
        )
        summary["decode_worker_tokens_per_second"] = expected / active["seconds"]
        summary["decode_worker_tokens_in_window"] = expected
        summary["decode_worker_window_seconds"] = active["seconds"]
        summary["throughput_definition"] = (
            "Complete per-request generation steps / elapsed first forward start to last token completion"
        )
        summary["arrival_requests_per_second"] = len(arrivals) / active["seconds"]
    else:
        summary["throughput_definition"] = (
            "Scheduled useful tokens between first and last iteration starts, excluding the final iteration"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((args.source / "case-registry.json").read_text())
    rows = []
    for isl, modes in registry.items():
        for mode, entry in modes.items():
            capacity_dir = args.source / entry["capacity_run"]
            capacity = json.loads((capacity_dir / "capacity-summary.json").read_text())
            maximum = capacity["maximum_sustained_concurrency"]
            probes = [t for t in capacity["trials"] if t["target"] == maximum and t["qualifies"]]
            selected = [("capacity", capacity_dir / t["result_file"]) for t in probes]
            for path in capacity_dir.glob("*common*results.json"):
                probe = json.loads(path.read_text())
                if probe["metrics"]["iterations_at_peak"] >= probe["output_tokens"] - 1:
                    selected.append(("matched", path))
            selected += [
                (
                    "serving",
                    args.source / entry["performance_run"] / "100-performance-1-results.json",
                )
            ]
            for kind, path in selected:
                name = f"{isl}-{mode}-{kind}-{path.stem}.json"
                target = args.output / name
                summary = json.loads(target.read_text()) if target.exists() else {}
                if summary.get("analysis_version") != 4:
                    summary = analyze(path)
                    summary["kind"] = kind
                    summary["maximum_sustained_concurrency"] = maximum
                    target.write_text(json.dumps(summary, indent=2) + "\n")
                row = {
                    k: v
                    for k, v in summary.items()
                    if k not in ("time_series", "partial_perf_active_window")
                }
                if "generation_active_intervals" in row:
                    row["generation_active_intervals"] = {
                        k: v
                        for k, v in row["generation_active_intervals"].items()
                        if k != "time_at_active_count_seconds"
                    }
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "isl": isl,
                            "mode": mode,
                            "kind": kind,
                            "requests": row["requests"],
                            "worker_tps": row["decode_worker_tokens_per_second"],
                            "cadence_ms": row["decode_step_cadence_ms"]["p50"],
                            "coverage": row["request_perf_coverage"],
                        }
                    ),
                    flush=True,
                )
                report = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "definition": (
                        "Serving: 159 generation tokens per completed request divided by elapsed time "
                        "from the first decode forward to the final decode token, using complete "
                        "per-request step records. Capacity probes with truncated per-request logs: "
                        "useful scheduled generation tokens between the first and last active "
                        "iteration start timestamps, excluding the last iteration's tokens. Sum "
                        "attention-DP ranks at the same iteration. Neither definition divides by "
                        "overlapping iterLatencyMS. One of the 160 output tokens was produced by prefill."
                    ),
                    "source": str(args.source),
                    "rows": rows,
                }
                temporary = args.output / "decode-throughput.tmp"
                temporary.write_text(json.dumps(report, indent=2) + "\n")
                temporary.replace(args.output / "decode-throughput.json")


if __name__ == "__main__":
    main()
