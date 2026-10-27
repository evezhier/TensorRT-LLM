#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-sparse-32k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1024G
#SBATCH --time=04:00:00
#SBATCH --exclusive
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO:?Set REPO to the absolute path of the benchmark checkout}"
: "${IMAGE:?Set IMAGE to the compatible container image}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent results directory}"

REPO=$(realpath -- "$REPO")
export REPO
repro_package="$REPO/examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1"
if [[ ! -f "$repro_package/run_slurm.sh" ]]; then
    echo "Repro scripts not found under REPO: $REPO" >&2
    exit 2
fi
if [[ -z "${MODEL_DIR:-}" ]]; then
    : "${LLM_MODELS_ROOT:?Set MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent}"
    MODEL_DIR="$LLM_MODELS_ROOT/DeepSeek-V4-Flash"
fi
MODEL_DIR=$(realpath -- "$MODEL_DIR")
LLM_MODELS_ROOT=$(dirname -- "$MODEL_DIR")
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
export MODEL_DIR LLM_MODELS_ROOT OUTPUT_ROOT

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "config.json is missing in checkpoint folder $MODEL_DIR" >&2
    exit 2
fi

if [[ -n "${WHEEL:-}" ]]; then
    WHEEL=$(realpath -- "$WHEEL")
    if [[ ! -f "$WHEEL" ]]; then
        echo "Prebuilt wheel not found: $WHEEL" >&2
        exit 2
    fi
    export WHEEL
fi

mkdir -p -- "$OUTPUT_ROOT"
export MOUNTS="${MOUNTS:-$REPO:$REPO,$LLM_MODELS_ROOT:$LLM_MODELS_ROOT,$OUTPUT_ROOT:$OUTPUT_ROOT}"
if [[ -n "${WHEEL:-}" ]]; then
    case "$WHEEL" in
        "$REPO"/*|"$LLM_MODELS_ROOT"/*|"$OUTPUT_ROOT"/*) ;;
        *)
            wheel_parent=$(dirname -- "$WHEEL")
            MOUNTS+=",$wheel_parent:$wheel_parent"
            ;;
    esac
fi
export JOB_ID="$SLURM_JOB_ID"
export CPUS="${SLURM_CPUS_PER_TASK:-64}"

bash "$repro_package/run_slurm.sh" serving "$OUTPUT_ROOT/32k-serving-$SLURM_JOB_ID"
