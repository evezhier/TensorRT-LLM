#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 2 || "$1" != "serving" && "$1" != "matched" ]]; then
    echo "Usage: bash reproduce_32k.sh {serving|matched} /absolute/fresh-output-directory"
    echo "Run inside a prepared container with 4 or 8 GPUs (serving) or 4 GPUs (matched)."
    echo "Required environment: MODEL_DIR or LLM_MODELS_ROOT. REPO defaults to this checkout."
    exit 2
fi
repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export REPO="${REPO:-$(realpath -- "$repro_package/../../../..")}"
if [[ -z "${MODEL_DIR:-}" ]]; then
    : "${LLM_MODELS_ROOT:?Set MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent}"
    MODEL_DIR="$LLM_MODELS_ROOT/DeepSeek-V4-Flash"
fi
MODEL_DIR=$(realpath -- "$MODEL_DIR")
LLM_MODELS_ROOT=$(dirname -- "$MODEL_DIR")
export MODEL_DIR LLM_MODELS_ROOT
repro_variant=$1
export OUTPUT
OUTPUT=$(realpath -m -- "$2")
if [[ -e "$OUTPUT" ]]; then
    echo "Output already exists; choose a fresh directory: $OUTPUT" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT"
cp -R -- "$repro_package/harness/." "$OUTPUT/"
if [[ "$repro_variant" == "matched" ]]; then
    cp -- "$OUTPUT/matched/benchmark_server.py" "$OUTPUT/benchmark_server.py"
    cp -- "$OUTPUT/matched/run_matched_cases.py" "$OUTPUT/run_matched_cases.py"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi
source "$OUTPUT/env.sh"
python3 "$OUTPUT/preflight.py" "$repro_variant" 2>&1 | tee "$OUTPUT/preflight.log"
nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,power.draw \
    --format=csv -l 1 > "$OUTPUT/gpu-telemetry.csv" &
repro_telemetry_pid=$!
cleanup_telemetry() {
    kill "$repro_telemetry_pid" 2>/dev/null || true
    wait "$repro_telemetry_pid" 2>/dev/null || true
}
trap cleanup_telemetry EXIT
if [[ "$repro_variant" == "serving" ]]; then
    python3 -u "$OUTPUT/run_ratios.py" | tee "$OUTPUT/controller.log"
    python3 "$OUTPUT/analyze_ratios.py" | tee "$OUTPUT/ratio-analysis.log"
    python3 "$OUTPUT/analyze_prefill_feed.py" | tee "$OUTPUT/prefill-feed.log"
else
    python3 -u "$OUTPUT/run_matched_cases.py" | tee "$OUTPUT/controller.log"
fi
python3 "$OUTPUT/analyze_fix.py" | tee "$OUTPUT/fix-analysis.log"
echo "Results: $OUTPUT/fix-analysis.json"
echo "Request completion and token parity are separate; inspect the parity entries."
