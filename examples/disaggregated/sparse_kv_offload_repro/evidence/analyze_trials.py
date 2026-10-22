# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize measured trials and exact token comparisons without model imports."""

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(values), "mean": statistics.mean(values),
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99),
            "min": min(values), "max": max(values)}


def output_counts(result: dict) -> Counter:
    return Counter(tuple(item["token_ids"]) for item in result["requests"] if item["passed"])


def compare(left: dict, right: dict) -> dict:
    a, b = output_counts(left), output_counts(right)
    shared = sum((a & b).values())
    return {
        "left_completed": sum(a.values()), "right_completed": sum(b.values()),
        "identical_token_multisets": a == b,
        "identical_output_set": set(a) == set(b),
        "exactly_matched_request_pairs": shared,
        "exactly_matched_output_tokens_per_mode": shared * left["output_tokens"],
        "left_unique_outputs": len(a), "right_unique_outputs": len(b),
    }


def compare_reference(result: dict, reference: list[int]) -> dict:
    differences = []
    compared = 0
    for request in result["requests"]:
        if not request["passed"]:
            continue
        compared += 1
        actual = request["token_ids"]
        first = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b), None)
        if first is None and len(actual) != len(reference):
            first = min(len(actual), len(reference))
        if first is not None:
            differences.append(first)
    return {"compared_requests": compared, "exact_reference_matches": compared - len(differences),
            "all_match_reference": compared > 0 and not differences,
            "first_mismatch_token_index_histogram": dict(sorted(Counter(differences).items()))}


def transfer_summary(records: list[dict], role: str) -> dict:
    workers = [record for record in records
               if f"perf_metrics-{role}-" in record.get("source_file", "")]
    transfers = []
    for record in workers:
        timing = record.get("perf_metrics", {}).get("timing_metrics", {})
        start, end = timing.get("kv_cache_transfer_start"), timing.get("kv_cache_transfer_end")
        size = timing.get("kv_cache_size")
        if start is not None and end is not None and start > 0 and end >= start:
            transfers.append({"request_id": record["request_id"], "start": start,
                              "end": end, "seconds": end - start, "bytes": size})
    sizes = [item["bytes"] for item in transfers if item["bytes"] is not None]
    interval = (max(item["end"] for item in transfers) - min(item["start"] for item in transfers)
                if transfers else None)
    return {
        "worker_records": len(workers),
        "timed_transfers": len(transfers),
        "receive_duration_seconds" if role == "generation" else "send_duration_seconds":
            distribution([item["seconds"] for item in transfers]),
        "payload_bytes_per_request": distribution(sizes),
        "total_payload_bytes": sum(sizes) if sizes else None,
        "observed_transfer_window_seconds": interval,
        "payload_bytes_per_observed_second": sum(sizes) / interval if sizes and interval else None,
        "rate_definition": (
            "Logical payload over the union's enclosing elapsed window; includes scheduling "
            "and queueing, not hardware link bandwidth"
        ),
        "transfers": transfers,
    }


def post_prefill_transfer_summary(records: list[dict]) -> dict:
    """Join same-host worker clocks at send readiness and receive completion."""
    contexts = {
        item.get("ctx_request_id"): item
        for item in records if "perf_metrics-context-" in item.get("source_file", "")
    }
    durations = []
    payloads = []
    for item in records:
        if "perf_metrics-generation-" not in item.get("source_file", ""):
            continue
        context = contexts.get(item.get("ctx_request_id"))
        if context is None:
            continue
        context_timing = context.get("perf_metrics", {}).get("timing_metrics", {})
        generation_timing = item.get("perf_metrics", {}).get("timing_metrics", {})
        start = context_timing.get("kv_cache_transfer_start")
        end = generation_timing.get("kv_cache_transfer_end")
        if start is not None and end is not None and start > 0 and end >= start:
            durations.append(end - start)
            if generation_timing.get("kv_cache_size") is not None:
                payloads.append(generation_timing["kv_cache_size"])
    return {
        "send_ready_to_receive_complete_seconds": distribution(durations),
        "payload_bytes_per_request": distribution(payloads),
        "definition": (
            "Context worker transfer start to generation worker transfer completion on the same host; "
            "includes transfer scheduling and excludes the preceding context queue and most prefill. "
            "This is an elapsed pipeline measurement, not an isolated DMA microbenchmark."
        ),
    }


def decode_gpu_summary(records: list[dict]) -> dict:
    """Deduplicate per-request copies of each rank's forward timing sample."""
    events = {}
    for record in records:
        if "perf_metrics-generation-" not in record.get("source_file", ""):
            continue
        steps = record.get("time_breakdown_metrics", {}).get("step_metrics", [])
        for step in steps:
            start = step.get("forward_start_time")
            end = step.get("forward_end_time")
            value = step.get("gpu_forward_time")
            if start is not None and end is not None and value is not None:
                events[(start, end)] = value
    return {"rank_forward_duration_ms": distribution(list(events.values())),
            "definition": "GPU forward event time; repeated per-request copies are deduplicated by forward timestamps"}


def native_transfer_summary(run: Path, result: dict) -> dict:
    """Select native transfer records by the measured trial's wall-clock interval."""
    ready = json.loads((run / "ready.json").read_text())
    zone = datetime.fromisoformat(ready["local_clock"]).tzinfo
    start = datetime.fromisoformat(result["started_at_utc"]).timestamp()
    end = datetime.fromisoformat(result["completed_at_utc"]).timestamp()
    sends = {}
    receives = {}
    for path in (run / "native-transfer").glob("*.csv"):
        with path.open() as stream:
            for row in csv.DictReader(stream):
                when = datetime.fromisoformat(row["timestamp"]).replace(tzinfo=zone).timestamp()
                if not start <= when <= end:
                    continue
                if "gen_side_transfer_time(ms)" in row:
                    receives[row["RequestID"]] = {
                        "milliseconds": float(row["gen_side_transfer_time(ms)"]),
                        "bytes": int(row["kv_cache_size"]),
                    }
                elif row.get("task_type") == "KVSendTask" and row.get("transfer_size_bytes"):
                    size = int(row["transfer_size_bytes"])
                    if size == 0:
                        continue
                    item = sends.setdefault(row["unique_rid"], {
                        "bytes": 0, "segments": 0, "transfer_ms": 0, "prepare_ms": 0, "queue_ms": 0,
                    })
                    item["bytes"] += size
                    item["segments"] += int(row["transfer_entry_count"])
                    for target, source in (("transfer_ms", "transfer_latency_ms"),
                                           ("prepare_ms", "prepare_args_latency_ms"),
                                           ("queue_ms", "queue_latency_ms")):
                        item[target] = max(item[target], float(row[source]))
    return {
        "send_request_count": len(sends), "receive_request_count": len(receives),
        "_selected_request_ids": list(sends),
        "expected_completed_requests": result["completed_requests"],
        "total_send_payload_bytes": sum(item["bytes"] for item in sends.values()),
        "send_payload_bytes_per_request": distribution([item["bytes"] for item in sends.values()]),
        "send_transfer_phase_ms": distribution([item["transfer_ms"] for item in sends.values()]),
        "send_prepare_phase_ms": distribution([item["prepare_ms"] for item in sends.values()]),
        "send_queue_phase_ms": distribution([item["queue_ms"] for item in sends.values()]),
        "segments_per_request": distribution([item["segments"] for item in sends.values()]),
        "receive_lifecycle_ms": distribution([item["milliseconds"] for item in receives.values()]),
        "send_phase_payload_mib_per_second": distribution([
            item["bytes"] / (item["transfer_ms"] / 1000) / 2**20
            for item in sends.values() if item["transfer_ms"] > 0
        ]),
        "definition": (
            "Native NIXL send-task elapsed transfer phase, excluding its preparation and queue phases. "
            "For a request with multiple source ranks, bytes are summed and phase times take the maximum. "
            "Receive lifecycle includes waiting for context. Rates are task measurements, not hardware link bandwidth."
        ),
    }


def summarize_trial(path: Path, reference: list[int] | None = None) -> dict:
    result = json.loads(path.read_text())
    summary = {key: value for key, value in result.items()
               if key not in ("requests", "output_histogram")}
    summary["result_file"] = str(path)
    native = native_transfer_summary(path.parent, result)
    selected_ids = set(native.pop("_selected_request_ids"))
    summary["native_transfer"] = native
    summary["token_output_hash_counts"] = {
        hashlib.sha256(json.dumps(tokens).encode()).hexdigest(): count
        for tokens, count in output_counts(result).items()
    }
    perf_path = path.with_name(path.name.replace("-results.json", "-perf.json"))
    if perf_path.exists():
        records = json.loads(perf_path.read_text())
        summary["auxiliary_perf_records_before_filter"] = len(records)
        records = [r for r in records if str(r.get("disagg_request_id")) in selected_ids]
        summary["auxiliary_perf_records_in_trial"] = len(records)
        summary["generation_transfer"] = transfer_summary(records, "generation")
        summary["context_transfer"] = transfer_summary(records, "context")
        summary["post_prefill_transfer"] = post_prefill_transfer_summary(records)
        summary["decode_gpu_forward"] = decode_gpu_summary(records)
    if reference is not None:
        summary["parity_to_resident_single_request"] = compare_reference(result, reference)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = None
    if args.reference:
        baseline = json.loads(args.reference.read_text())
        reference = baseline["requests"][0]["token_ids"]
    trials = [summarize_trial(path, reference)
              for path in sorted(args.run.glob("*-results.json"))]
    (args.run / "trial-summary.json").write_text(json.dumps(trials, indent=2) + "\n")
    for trial in trials:
        print(json.dumps({key: trial[key] for key in (
            "result_file", "completed_requests", "seconds", "output_tokens_per_second", "unique_outputs"
        )} | {"active_peak": trial["metrics"]["peak_concurrent_decode"]}))


if __name__ == "__main__":
    main()
