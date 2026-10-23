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
    for path in sorted(output.glob("p*_t*_*.json")):
        case = json.loads(path.read_text())
        if not case["all_samples_verified"]:
            raise RuntimeError(f"Unverified transfer in {path}")
        rows.append(
            {
                "case": path.stem,
                "mode": case["mode"],
                "segments": case["segments"],
                "payload_bytes": case["payload_bytes"],
                "registered_bytes": case["registered_bytes"],
                "num_threads": case["num_threads"],
                "ucx_tls": case["ucx_tls"],
                "transfer_p50_ms": case["transfer_ms"]["p50"],
                "transfer_p95_ms": case["transfer_ms"]["p95"],
                "local_copy_p50_ms": case["local_copy_ms"]["p50"],
                "total_p50_ms": case["total_ms"]["p50"],
                "total_p95_ms": case["total_ms"]["p95"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed probe cases in {output}")
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
