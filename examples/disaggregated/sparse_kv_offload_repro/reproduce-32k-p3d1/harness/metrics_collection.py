# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared metric summaries and completion-aware draining for benchmark clients."""

import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime

import requests


def distribution(values: list[float]) -> dict:
    """Summarize measurements without replacing missing observations with zero."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": min(values),
        "max": max(values),
    }


def worker_metrics(workers: dict) -> dict:
    """Drain executor iteration statistics from both serving roles."""
    result = {}
    for role in ("context_servers", "generation_servers"):
        result[role] = []
        for worker_index, worker in enumerate(workers[role]):
            response = requests.get(
                f"http://{worker['host']}:{worker['port']}/metrics", timeout=120
            )
            response.raise_for_status()
            result[role].extend(
                {**row, "worker_index": worker_index, "worker_port": worker["port"]}
                for row in response.json()
            )
    return result


def summarize_metrics(stats: dict) -> dict:
    """Sum attention-DP ranks at the same iteration, preserving occupancy details."""
    iterations = defaultdict(dict)
    per_rank = defaultdict(int)
    memory = defaultdict(int)
    context_shapes = Counter()
    paused = 0
    queue_peak = 0
    durations = []
    kv_peaks = defaultdict(float)
    for role, rows in stats.items():
        for row in rows:
            batching = row["inflightBatchingStats"]
            rank = row.get("attentionDpRank", row.get("rank", 0))
            memory[f"{role}:{rank}"] = max(memory[f"{role}:{rank}"], row.get("gpuMemUsage", 0))
            if role == "generation_servers":
                count = batching["numGenRequests"]
                iterations[row["iter"]][rank] = count
                per_rank[rank] = max(per_rank[rank], count)
                paused = max(paused, batching.get("numPausedRequests", 0))
                queue_peak = max(queue_peak, row.get("numQueuedRequests", 0))
                if count and "iterLatencyMS" in row:
                    durations.append(row["iterLatencyMS"])
                for key, value in row.get("kvCacheStats", {}).items():
                    if isinstance(value, (int, float)):
                        kv_peaks[f"rank{rank}:{key}"] = max(kv_peaks[f"rank{rank}:{key}"], value)
            elif batching["numScheduledRequests"]:
                context_shapes[
                    (
                        batching["numContextRequests"],
                        batching["numCtxTokens"],
                        batching["numCtxKvTokens"],
                    )
                ] += 1
    histogram = Counter(sum(ranks.values()) for ranks in iterations.values())
    peak = max(histogram, default=0)
    return {
        "peak_concurrent_decode": peak,
        "iterations_at_peak": histogram.get(peak, 0),
        "peak_decode_per_rank": dict(per_rank),
        "decode_concurrency_histogram": dict(sorted(histogram.items())),
        "gpu_memory_bytes": dict(memory),
        "max_paused_requests_per_rank": paused,
        "max_queued_requests": queue_peak,
        "generation_kv_stat_peaks": dict(kv_peaks),
        "decode_iteration_latency_ms": distribution(durations),
        "context_schedule_histogram": [
            {"requests": key[0], "tokens": key[1], "kv_tokens": key[2], "iterations": count}
            for key, count in sorted(context_shapes.items())
        ],
    }


def completed_requests(stats: dict) -> int:
    rows = {
        (r.get("attentionDpRank", r.get("rank", 0)), r["iter"]): r
        for r in stats.get("generation_servers", [])
    }
    return sum(r.get("numCompletedRequests", 0) for r in rows.values())


def scoped_metrics(stats: dict, started_at: str, completed_at: str, local_clock: str) -> dict:
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(completed_at)
    timezone = datetime.fromisoformat(local_clock).tzinfo
    return {
        role: [
            r
            for r in rows
            if start
            <= datetime.strptime(r["timestamp"], "%m-%d-%Y %H:%M:%S.%f").replace(tzinfo=timezone)
            <= end
        ]
        for role, rows in stats.items()
    }


def wake_stats(serve_url: str, model: str, prompt: list[int]) -> None:
    response = requests.post(
        serve_url + "/v1/completions",
        json={
            "model": model,
            "prompt": prompt[:128],
            "max_tokens": 1,
            "min_tokens": 1,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "detokenize": False,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()
    if "error" in response.json():
        raise RuntimeError(str(response.json()["error"]))


def finish_metrics(
    workers: dict,
    stats: dict,
    expected: int,
    timeout: float = 30,
    scope: tuple | None = None,
    wake: tuple | None = None,
) -> dict:
    # Completion counters are populated only on attention-DP rank zero.
    expected_rank0 = max(1, expected // 2) if expected else 0
    if scope:
        stats = scoped_metrics(stats, *scope)
    if completed_requests(stats) < expected_rank0 and wake:
        wake_stats(*wake)
    deadline = time.monotonic() + timeout
    while completed_requests(stats) < expected_rank0 and time.monotonic() < deadline:
        time.sleep(0.5)
        extra = worker_metrics(workers)
        if scope:
            extra = scoped_metrics(extra, *scope)
        for role, rows in extra.items():
            stats.setdefault(role, []).extend(rows)
    return stats
