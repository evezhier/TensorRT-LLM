#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-sparse-32k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=160
#SBATCH --mem=1024G
#SBATCH --time=04:00:00
#SBATCH --exclusive
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${E2E_REPO:?Set E2E_REPO to the absolute path of the benchmark checkout}"
: "${REPRO_IMAGE:?Set REPRO_IMAGE to the compatible container image}"
: "${REPRO_OUTPUT_ROOT:?Set REPRO_OUTPUT_ROOT to a persistent results directory}"

E2E_REPO=$(realpath -- "$E2E_REPO")
export E2E_REPO
repro_package="$E2E_REPO/examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1"
if [[ ! -f "$repro_package/run_slurm.sh" ]]; then
    echo "Repro scripts not found under E2E_REPO: $E2E_REPO" >&2
    exit 2
fi
if [[ -z "${REPRO_MODEL_DIR:-}" ]]; then
    : "${LLM_MODELS_ROOT:?Set REPRO_MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent}"
    REPRO_MODEL_DIR="$LLM_MODELS_ROOT/DeepSeek-V4-Flash"
fi
REPRO_MODEL_DIR=$(realpath -- "$REPRO_MODEL_DIR")
LLM_MODELS_ROOT=$(dirname -- "$REPRO_MODEL_DIR")
REPRO_OUTPUT_ROOT=$(realpath -m -- "$REPRO_OUTPUT_ROOT")
export REPRO_MODEL_DIR LLM_MODELS_ROOT REPRO_OUTPUT_ROOT

if [[ ! -f "$REPRO_MODEL_DIR/config.json" ]]; then
    echo "config.json is missing in checkpoint folder $REPRO_MODEL_DIR" >&2
    exit 2
fi

if [[ -n "${REPRO_WHEEL:-}" ]]; then
    REPRO_WHEEL=$(realpath -- "$REPRO_WHEEL")
    if [[ ! -f "$REPRO_WHEEL" ]]; then
        echo "Prebuilt wheel not found: $REPRO_WHEEL" >&2
        exit 2
    fi
    export REPRO_WHEEL
fi

mkdir -p -- "$REPRO_OUTPUT_ROOT"
export REPRO_MOUNTS="${REPRO_MOUNTS:-$E2E_REPO:$E2E_REPO,$LLM_MODELS_ROOT:$LLM_MODELS_ROOT,$REPRO_OUTPUT_ROOT:$REPRO_OUTPUT_ROOT}"
if [[ -n "${REPRO_WHEEL:-}" ]]; then
    case "$REPRO_WHEEL" in
        "$E2E_REPO"/*|"$LLM_MODELS_ROOT"/*|"$REPRO_OUTPUT_ROOT"/*) ;;
        *)
            wheel_parent=$(dirname -- "$REPRO_WHEEL")
            REPRO_MOUNTS+=",$wheel_parent:$wheel_parent"
            ;;
    esac
fi
export REPRO_JOB_ID="$SLURM_JOB_ID"
export REPRO_CPUS="${SLURM_CPUS_PER_TASK:-160}"

bash "$repro_package/run_slurm.sh" serving "$REPRO_OUTPUT_ROOT/32k-serving-$SLURM_JOB_ID"
