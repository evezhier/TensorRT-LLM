# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare direct and agent-bounce 32K offload runs on the same GPU allocation."""

import csv
import json
import re
from pathlib import Path

from analyze_transport_accuracy import normalize_config
from analyze_trials import native_transfer_summary
from run_ratios import publish


def analyze(root: Path) -> dict:
    preflight = json.loads((root / "preflight.json").read_text())
    ratio = preflight["prefill_workers"]
    cases = (f"offload-direct-p{ratio}-d1", f"offload-agent-p{ratio}-d1")
    rows = []
    configs = []
    for name in cases:
        case = root / "cases" / name
        config = json.loads((case / "config.json").read_text())
        result = json.loads((case / "100-performance-1-results.json").read_text())
        native = native_transfer_summary(case, result)
        host_counts = [
            int(value)
            for value in re.findall(
                r"bounce_host_submit_count=(\d+)", (root / "cases" / f"{name}.log").read_text()
            )
        ]
        configs.append(config)
        rows.append(
            {
                "run": name,
                "requests": result["command"]["requests"],
                "completed": result["completed_requests"],
                "all_requests_passed": result["all_requests_passed"],
                "send_records": native["send_request_count"],
                "receive_records": native["receive_request_count"],
                "payload_bytes": native["total_send_payload_bytes"],
                "send_p50_ms": native["send_transfer_phase_ms"]["p50"],
                "send_p95_ms": native["send_transfer_phase_ms"]["p95"],
                "receive_p50_ms": native["receive_lifecycle_ms"]["p50"],
                "receive_p95_ms": native["receive_lifecycle_ms"]["p95"],
                "output_tokens_per_second": result["output_tokens_per_second"],
                "e2e_p50_seconds": result["latency_seconds"]["p50"],
                "e2e_p95_seconds": result["latency_seconds"]["p95"],
                "ttft_p50_seconds": result["ttft_seconds"]["p50"],
                "host_bounce_submissions": sum(host_counts),
            }
        )
    configs_match = normalize_config(configs[0]) == normalize_config(configs[1])
    direct_disabled = all(
        not configs[0][key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
        for key in ("context_config", "generation_config")
    )
    agent_enabled = all(
        configs[1][key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
        for key in ("context_config", "generation_config")
    )
    complete = all(
        row["all_requests_passed"] and row["completed"] == row["requests"] == 4096 for row in rows
    )
    timings_complete = all(
        row["send_records"] == row["receive_records"] == row["completed"]
        and row["send_p50_ms"] is not None
        for row in rows
    )
    payloads_match = rows[0]["payload_bytes"] == rows[1]["payload_bytes"] > 0
    if not configs_match or not direct_disabled or not agent_enabled:
        status = "config_mismatch"
    elif not complete:
        status = "requests_incomplete"
    elif not timings_complete:
        status = "timing_incomplete"
    elif not payloads_match:
        status = "payload_mismatch"
    elif rows[1]["host_bounce_submissions"] == 0:
        status = "host_bounce_not_engaged"
    else:
        status = "pass"
    return {
        "status": status,
        "configs_match_except_transport": configs_match,
        "payloads_match": payloads_match,
        "send_p50_speedup": (
            rows[0]["send_p50_ms"] / rows[1]["send_p50_ms"]
            if rows[0]["send_p50_ms"] and rows[1]["send_p50_ms"]
            else None
        ),
        "preflight": {
            key: preflight.get(key)
            for key in ("head", "wheel_sha256", "model_config_sha256", "gpus", "ucx_tls")
        },
        "runs": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    report = analyze(root)
    publish(root / "transport-performance.json", report)
    with (root / "transport-performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report["runs"][0].keys())
        writer.writeheader()
        writer.writerows(report["runs"])
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
