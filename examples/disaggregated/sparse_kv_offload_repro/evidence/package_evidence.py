# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package source changes and completed validation records for the report."""

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    names = [
        "final-report.json", "fix-analysis.json", "ratio-analysis.json",
        "matched-decode-metrics.csv", "ratio-metrics.csv", "test-summary.json",
        "binary-manifest.json", "fix-only.patch", "fix-patch-summary.json",
        "local-before.patch", "local-after.patch", "pool-geometry-comparison.json",
        "serving-pool-geometry-comparison.json",
        "asynchronous-parity-detail.json", "preview-validation.json", "prefill-feed.json",
        "fix-visualization-data.json", "fix-comparison.html", "cleanup.json",
        "fallback-validation-status.json", "regressions.xml", "offload-metadata.xml",
        "analyze_fix.py", "analyze_ratios.py", "analyze_decode.py", "analyze_prefill_feed.py",
        "analyze_trials.py", "build_final_report.py", "build_fix_visualization.py",
        "benchmark_server.py", "run_ratios.py", "run_matched_cases.py",
        "package_evidence.py",
    ]
    paths = [ROOT / name for name in names]
    paths.extend(ROOT.glob("native-*.xml"))
    for cases in (ROOT / "cases", ROOT / "fallback-p1/cases"):
        for run in sorted(cases.iterdir()):
            if not run.is_dir() or not (run / "finished.json").exists():
                continue
            for name in ("config.json", "ready.json", "finished.json"):
                paths.append(run / name)
            for pattern in ("*-results.json", "sparse-counters-rank*.json", "memory-allocation-rank*.json"):
                paths.extend(run.glob(pattern))
    paths = sorted(set(paths))
    assert all(path.is_file() for path in paths)
    records = []
    target = ROOT / "residency-fix-evidence.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in paths:
            data = path.read_bytes()
            relative = str(path.relative_to(ROOT))
            original = {"source_file": str(path), "source_bytes": len(data),
                        "source_sha256": hashlib.sha256(data).hexdigest()}
            if path.name.endswith("-results.json"):
                value = json.loads(data)
                value.pop("metrics")
                for request in value["requests"]:
                    request.pop("token_arrival_seconds")
                    request.pop("perf_headers")
                value["evidence_projection"] = {
                    "omitted": ["metrics", "requests[].token_arrival_seconds", "requests[].perf_headers"],
                    "retained": "Every request ID, exact token sequence, pass flag and latency summary; all top-level fields except repeated raw metrics",
                    **original,
                }
                data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
                relative = relative.replace("-results.json", "-client-evidence.json")
            records.append({"file": relative, "bytes": len(data),
                            "sha256": hashlib.sha256(data).hexdigest(), **original})
            bundle.writestr(relative, data)
        manifest = {"scope": "32K sparse GPU residency fix and completed validation",
                    "excluded": "Raw server logs, environment dumps, model weights and large raw GPU/per-request performance logs",
                    "files": records}
        bundle.writestr("evidence-manifest.json", json.dumps(manifest, indent=2) + "\n")
    assert target.stat().st_size <= 10 * 1024 * 1024
    with zipfile.ZipFile(target) as bundle:
        assert bundle.testzip() is None
    summary = {"file": str(target), "files": len(records) + 1,
               "bytes": target.stat().st_size,
               "sha256": hashlib.file_digest(target.open("rb"), "sha256").hexdigest()}
    (ROOT / "evidence-bundle.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
