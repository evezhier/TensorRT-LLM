# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare matched direct and agent-bounce offload outputs without starting servers."""

import argparse
import copy
import json
import re
from pathlib import Path

from analyze_trials import compare


def normalize_config(config: dict) -> dict:
    """Remove per-run paths and transport selection from a serving config."""
    value = copy.deepcopy(config)
    value["disaggregation_config"].pop("perf_metrics_output_dir", None)
    for key in ("context_config", "generation_config"):
        value[key].pop("perf_metrics_output_dir", None)
        transport = value[key]["cache_transceiver_config"]
        for field in (
            "kv_cache_bounce_size_mb",
            "agent_bounce_buffer_enable",
            "agent_bounce_params",
        ):
            transport.pop(field, None)
    return value


def analyze(root: Path) -> dict:
    cases = (
        "offload-direct-a-matched-p1-d1",
        "offload-direct-b-matched-p1-d1",
        "offload-agent-matched-p1-d1",
    )
    trials = ("000-cold", "001-warm", "100-c1", "101-c96")
    comparisons = []
    for trial in trials:
        results = [
            json.loads((root / "cases" / name / f"{trial}-results.json").read_text())
            for name in cases
        ]
        comparisons.append(
            {
                "trial": trial,
                "direct_repeat": compare(results[0], results[1]),
                "direct_to_agent": compare(results[0], results[2]),
            }
        )

    configs = [json.loads((root / "cases" / name / "config.json").read_text()) for name in cases]
    normalized = [normalize_config(config) for config in configs]
    configs_match = normalized[0] == normalized[1] == normalized[2]
    direct_disabled = all(
        not config[key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
        for config in configs[:2]
        for key in ("context_config", "generation_config")
    )
    agent_enabled = all(
        configs[2][key]["cache_transceiver_config"].get("agent_bounce_buffer_enable")
        for key in ("context_config", "generation_config")
    )
    agent_log = root / "cases" / f"{cases[2]}.log"
    submits = [
        int(value) for value in re.findall(r"bounce_submit_count=(\d+)", agent_log.read_text())
    ]
    baseline_stable = all(row["direct_repeat"]["identical_token_multisets"] for row in comparisons)
    agent_matches = all(row["direct_to_agent"]["identical_token_multisets"] for row in comparisons)
    if not configs_match or not direct_disabled:
        status = "config_mismatch"
    elif not baseline_stable:
        status = "baseline_unstable"
    elif not agent_matches:
        status = "agent_mismatch"
    elif not agent_enabled or not any(count > 0 for count in submits):
        status = "agent_not_engaged"
    else:
        status = "pass"
    report = {
        "status": status,
        "configs_match_except_transport": configs_match,
        "direct_agent_bounce_disabled": direct_disabled,
        "agent_enabled": agent_enabled,
        "agent_bounce_submit_counts": submits,
        "comparisons": comparisons,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = analyze(args.run)
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
