# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package verified residency-fix results with their scope and source provenance."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "latest-p3d1-nsys-20261018T232117Z"


def read(name: str) -> dict | list:
    return json.loads((ROOT / name).read_text())


def main() -> None:
    analysis = read("fix-analysis.json")
    runs = [run for run in analysis["runs"] if "fallback-p1" in run["directory"]]
    assert len(runs) == 2 and all(run["finished"] for run in runs)
    serving_runs = [run for run in analysis["runs"] if run not in runs]
    assert len(serving_runs) == 2 and all(run["finished"] for run in serving_runs)
    assert all(trial["all_requests_passed"] for run in serving_runs for trial in run["trials"])
    serving = read("ratio-analysis.json")
    rows = []
    for run in runs:
        for trial in run["trials"]:
            assert trial["all_requests_passed"]
            assert trial["requests"] == trial["command"]["requests"]
            if trial["command"]["gate"]:
                concurrency = trial["command"]["concurrency"]
                assert trial["iteration_token_coverage_complete"]
                assert trial["decode_iterations"] == 159
                assert trial["scheduled_decode_count"]["min"] == concurrency
                assert trial["scheduled_decode_count"]["max"] == concurrency
                assert len(trial["decode_start_barrier"]) == 2
                assert all(gate["passed"] and gate["completed"] == concurrency
                           for gate in trial["decode_start_barrier"])
                rows.append({
                    "mode": run["config"]["mode"], "trial": trial["trial"],
                    "active_decode_requests": concurrency,
                    "decode_tokens_per_second": trial["scheduled_tokens_per_iteration_window_second"],
                    "gpu_forward_ms_p50": trial["generation_forward_event_ms"]["p50"],
                    "decode_cadence_ms_p50": trial["decode_step_cadence_ms"]["p50"],
                })
    controlled = [p for p in analysis["parity"]
                  if "fallback-p1" in p["right"] and "200-churn" not in p["right"]]
    assert len(controlled) == 7 and all(p["identical_token_multisets"] for p in controlled)
    offload = next(run for run in runs if run["config"]["mode"] == "offload")
    serving_offload = next(run for run in serving_runs if run["config"]["mode"] == "offload")
    assert len(offload["sparse_counters"]) == 2
    assert len(serving_offload["sparse_counters"]) == 2
    assert offload["total_sparse_hits"] > 0 and offload["total_sparse_misses"] > 0
    previous = json.loads((OLD / "final-report.json").read_text())
    previous_matched = [json.loads(path.read_text()) for path in
                        (OLD / "cases/offload-nsys-matched-p3-d1").glob("sparse-counters-rank*.json")]
    previous_hits = sum(row["hits"] for row in previous_matched)
    previous_misses = sum(row["misses"] for row in previous_matched)
    tests = read("test-summary.json")
    assert all(int(suite["errors"]) == int(suite["failures"]) == 0 for suite in tests.values())
    source_names = [
        "fix-analysis.json", "test-summary.json", "binary-manifest.json", "fix-only.patch",
        "local-before.patch", "local-after.patch", "pool-geometry-comparison.json",
        "serving-pool-geometry-comparison.json",
        "asynchronous-parity-detail.json", "preview-validation.json", "fix-visualization-data.json",
        "cleanup.json", "ratio-analysis.json", "ratio-metrics.csv", "prefill-feed.json",
    ]
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "32K full-model validation of persistent sparse GPU cache residency, matched batches and 3:1 serving",
        "page_url": "https://chatgpt.com/space/page_bec74a2e48208191ade22eaebdb248de",
        "revision": read("binary-manifest.json"),
        "fix": {
            "cause": "The Python adapter removed every native Batch member after each forward, releasing GPU sparse-cache leases and clearing residency at the next attachment.",
            "behavior": "Continuously scheduled requests retain native Batch membership. Only departing requests detach. Native reorder moves row bindings and flush watermarks without releasing leases; legacy host page tables receive snapshots.",
            "regressions": "Request additions, removals, permutations, row compaction, stable device addresses, page-table snapshots and sparse KV data plus NVFP4 scales.",
            "isolated_patch": read("fix-patch-summary.json"),
        },
        "settings": {
            "input_tokens": 32768, "output_tokens": 160, "model_layers": 43,
            "matched_prefill_workers": 1, "serving_prefill_workers": 3,
            "decode_workers": 1, "gpus_per_worker": 2,
            "decoder_attention_dp": 2, "speculative_decoding": False,
            "gpu_cache_factor": 2.0, "host_budget_gib_per_rank": 256,
            "matched_node": "umb-b200-247", "matched_job_id": 5115369, "matched_cpus": 96,
            "serving_node": "umb-b200-260", "serving_job_id": 5115013, "serving_cpus": 160,
            "prompt": "Identical synthetic astronomy text; prefill prefix reuse enabled, decoder prefix reuse disabled",
        },
        "validation": {
            "test_suites": tests, "full_model_completed_requests": sum(t["requests"] for run in analysis["runs"] for t in run["trials"]),
            "matched_ready_batch_token_parity_passed": True,
            "matched_ready_batch_sizes": [1, 96, 144, 1204],
            "offload_1696_repeat_token_parity_passed": True,
            "asynchronous_parity": read("asynchronous-parity-detail.json"),
        },
        "matched_decode_metrics": rows,
        "matched_throughput_definition": "Useful scheduled tokens over 158 complete intervals between the 159 decode iteration starts; includes decode scheduling cadence, excludes prefill and KV transfer before the release gate.",
        "matched_sparse_cache": {
            "hits": offload["total_sparse_hits"], "misses": offload["total_sparse_misses"],
            "hit_rate_percent": offload["sparse_hit_rate_percent"],
            "scope": analysis["counter_scope"],
            "logical_bytes_per_selected_unit_including_nvfp4_scales": 324,
            "logical_miss_payload_bytes": offload["total_sparse_misses"] * 324,
            "counter_sources": offload["sparse_counters"],
            "historical_serving_hit_rate_percent": 100 * previous["sparse_logical_traffic"]["hit_ratio"],
            "historical_matched_hit_rate_percent": 100 * previous_hits / (previous_hits + previous_misses),
            "historical_comparison_scope": "Before-fix counters from the earlier 3:1 serving and matched-batch processes; trial mixes differ from the new validation process.",
        },
        "serving_sparse_cache": {
            "hits": serving_offload["total_sparse_hits"],
            "misses": serving_offload["total_sparse_misses"],
            "hit_rate_percent": serving_offload["sparse_hit_rate_percent"],
            "scope": "Whole 3:1 offload worker process, including cold, warm and 128-request warmup trials plus 4096 measured requests",
            "counter_sources": serving_offload["sparse_counters"],
        },
        "pool_geometry_comparison": read("pool-geometry-comparison.json"),
        "serving_pool_geometry_comparison": read("serving-pool-geometry-comparison.json"),
        "runs": analysis["runs"], "parity": analysis["parity"],
        "serving": serving,
        "prefill_feed": read("prefill-feed.json"),
        "historical_serving": previous["serving"],
        "baseline_reference": str(OLD / "final-report.json"),
        "limitations": [
            "These new full-model measurements use 1:1 prefill:decode for matched batches and 3:1 for serving, both at 32K. The 128K and 1M measurements precede this residency fix.",
            "The initial 384 GiB/rank host-budget configuration was rejected by the smaller allocation. The successful 256 GiB/rank configuration allocates exactly the same physical sparse arena, history pools and GPU quota as the earlier matched offload run.",
            "Exact token multisets match at controlled ready batches and across the 1696-request repeat. The 1:1 asynchronous arrivals matched 508 of 512 output sequences; every offload output sequence also appeared in resident mode. The separate 3:1 serving parity is recorded in serving.parity.",
            "The selected 1204 and 1696 batches validate previously established capacities; this is not a new maximum-capacity search.",
            "No GPU profiler was active in these validation measurements. Earlier profiler measurements used another node and CPU allocation and are historical references, not a controlled estimate of speedup.",
        ],
        "cleanup": read("cleanup.json"),
        "source_files": [
            {"file": str(ROOT / name), "bytes": (ROOT / name).stat().st_size,
             "sha256": hashlib.file_digest((ROOT / name).open("rb"), "sha256").hexdigest()}
            for name in source_names
        ],
    }
    (ROOT / "final-report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (ROOT / "matched-decode-metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"requests": report["validation"]["full_model_completed_requests"],
                      "matched_hit_rate_percent": report["matched_sparse_cache"]["hit_rate_percent"],
                      "serving_hit_rate_percent": report["serving_sparse_cache"]["hit_rate_percent"],
                      "rows": rows}))


if __name__ == "__main__":
    main()
