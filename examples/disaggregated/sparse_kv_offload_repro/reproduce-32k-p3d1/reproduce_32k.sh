#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 || "$1" != "serving" && "$1" != "matched" ]]; then
    echo "Usage: bash reproduce_32k.sh {serving|matched} /absolute/fresh-output-directory"
    echo "Run inside a prepared container with 8 GPUs (serving) or 4 GPUs (matched)."
    echo "Required environment: REPRO_MODEL_DIR or LLM_MODELS_ROOT. E2E_REPO defaults to this checkout."
    exit 2
fi
repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export E2E_REPO="${E2E_REPO:-$(realpath -- "$repro_package/../../../..")}"
if [[ -z "${REPRO_MODEL_DIR:-}" ]]; then
    : "${LLM_MODELS_ROOT:?Set REPRO_MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent}"
    REPRO_MODEL_DIR="$LLM_MODELS_ROOT/DeepSeek-V4-Flash"
fi
REPRO_MODEL_DIR=$(realpath -- "$REPRO_MODEL_DIR")
LLM_MODELS_ROOT=$(dirname -- "$REPRO_MODEL_DIR")
export REPRO_MODEL_DIR LLM_MODELS_ROOT
repro_variant=$1
export REPRO_OUTPUT
REPRO_OUTPUT=$(realpath -m -- "$2")
if [[ -e "$REPRO_OUTPUT" ]]; then
    echo "Output already exists; choose a fresh directory: $REPRO_OUTPUT" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$REPRO_OUTPUT")"
mkdir -- "$REPRO_OUTPUT"
cp -R -- "$repro_package/harness/." "$REPRO_OUTPUT/"
if [[ "$repro_variant" == "matched" ]]; then
    cp -- "$REPRO_OUTPUT/matched/benchmark_server.py" "$REPRO_OUTPUT/benchmark_server.py"
    cp -- "$REPRO_OUTPUT/matched/run_matched_cases.py" "$REPRO_OUTPUT/run_matched_cases.py"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi
source "$REPRO_OUTPUT/env.sh"
python3 "$REPRO_OUTPUT/preflight.py" "$repro_variant" > "$REPRO_OUTPUT/preflight.log" 2>&1
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw \
    --format=csv -l 1 > "$REPRO_OUTPUT/gpu-telemetry.csv" &
repro_telemetry_pid=$!
cleanup_telemetry() {
    kill "$repro_telemetry_pid" 2>/dev/null || true
    wait "$repro_telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT
if [[ "$repro_variant" == "serving" ]]; then
    python3 -u "$REPRO_OUTPUT/run_ratios.py" | tee "$REPRO_OUTPUT/controller.log"
    python3 "$REPRO_OUTPUT/analyze_ratios.py" | tee "$REPRO_OUTPUT/ratio-analysis.log"
    python3 "$REPRO_OUTPUT/analyze_prefill_feed.py" | tee "$REPRO_OUTPUT/prefill-feed.log"
else
    python3 -u "$REPRO_OUTPUT/run_matched_cases.py" | tee "$REPRO_OUTPUT/controller.log"
fi
python3 "$REPRO_OUTPUT/analyze_fix.py" | tee "$REPRO_OUTPUT/fix-analysis.log"
echo "Results: $REPRO_OUTPUT/fix-analysis.json"
echo "Request completion and token parity are separate; inspect the parity entries."
