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
: "${REPRO_IMAGE:?Set REPRO_IMAGE to the compatible container image}"
: "${REPRO_OUTPUT_ROOT:?Set REPRO_OUTPUT_ROOT to a persistent results directory}"

repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export E2E_REPO
E2E_REPO=$(git -C "$repro_package" rev-parse --show-toplevel)
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

mkdir -p -- "$REPRO_OUTPUT_ROOT"
export REPRO_MOUNTS="${REPRO_MOUNTS:-$E2E_REPO:$E2E_REPO,$LLM_MODELS_ROOT:$LLM_MODELS_ROOT,$REPRO_OUTPUT_ROOT:$REPRO_OUTPUT_ROOT}"
export REPRO_JOB_ID="$SLURM_JOB_ID"
export REPRO_CPUS="${SLURM_CPUS_PER_TASK:-160}"

bash "$repro_package/run_slurm.sh" serving "$REPRO_OUTPUT_ROOT/32k-serving-$SLURM_JOB_ID"
