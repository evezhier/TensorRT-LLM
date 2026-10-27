#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize byte-verified host-transfer probe cases."""

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    output = Path(sys.argv[1])
    rows = []
    for path in sorted(output.glob("**/p*_t*_*.json")):
        case = json.loads(path.read_text())
        if not case["all_samples_verified"]:
            raise RuntimeError(f"Unverified transfer in {path}")
        case_path = path.relative_to(output)
        rows.append(
            {
                "case": str(case_path.with_suffix("")),
                "case_key": path.stem,
                "ucx_sweep": case.get(
                    "ucx_sweep", case_path.parent.name if case_path.parent != Path(".") else "baseline"
                ),
                "mode": case["mode"],
                "segments": case["segments"],
                "payload_bytes": case["payload_bytes"],
                "registered_bytes": case["registered_bytes"],
                "num_threads": case["num_threads"],
                "ucx_tls": case["ucx_tls"],
                "ucx_rndv_scheme": case.get("ucx_rndv_scheme"),
                "ucx_cuda_ipc_enable_mnnvl": case.get("ucx_cuda_ipc_enable_mnnvl"),
                "sender_log": str(case_path.with_suffix(".sender.log")),
                "transfer_p50_ms": case["transfer_ms"]["p50"],
                "transfer_p95_ms": case["transfer_ms"]["p95"],
                "local_copy_p50_ms": case["local_copy_ms"]["p50"],
                "total_p50_ms": case["total_ms"]["p50"],
                "total_p95_ms": case["total_ms"]["p95"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed probe cases in {output}")
    baseline = {row["case_key"]: row for row in rows if row["ucx_sweep"] == "baseline"}
    for row in rows:
        reference = baseline.get(row["case_key"])
        row["transfer_speedup_vs_baseline"] = (
            reference["transfer_p50_ms"] / row["transfer_p50_ms"]
            if reference is not None and row["transfer_p50_ms"] > 0
            else None
        )
        row["total_speedup_vs_baseline"] = (
            reference["total_p50_ms"] / row["total_p50_ms"]
            if reference is not None and row["total_p50_ms"] > 0
            else None
        )
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['case']}: NIXL p50={row['transfer_p50_ms']:.3f} ms, "
            f"copy p50={row['local_copy_p50_ms']:.3f} ms, "
            f"sum p50={row['total_p50_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
