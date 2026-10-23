#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=sparse-host-transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO:?Set REPO to the benchmark checkout}"
: "${IMAGE:?Set IMAGE to the container image or squashfs}"
: "${WHEEL:?Set WHEEL to the built TensorRT-LLM wheel}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent results directory}"

REPO=$(realpath -- "$REPO")
WHEEL=$(realpath -- "$WHEEL")
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
probe="$REPO/examples/disaggregated/sparse_kv_offload_repro/host_transfer_probe"
if [[ ! -f "$probe/run.sh" || ! -f "$REPO/requirements-dev.txt" ]]; then
    echo "Probe or development requirements missing under REPO: $REPO" >&2
    exit 2
fi
if [[ ! -f "$WHEEL" ]]; then
    echo "Wheel not found: $WHEEL" >&2
    exit 2
fi
mkdir -p -- "$OUTPUT_ROOT"
export REPO WHEEL OUTPUT="$OUTPUT_ROOT/host-transfer-$SLURM_JOB_ID"
MOUNTS="${MOUNTS:-$REPO:$REPO,$OUTPUT_ROOT:$OUTPUT_ROOT}"
case "$WHEEL" in
    "$REPO"/*|"$OUTPUT_ROOT"/*) ;;
    *) wheel_parent=$(dirname -- "$WHEEL"); MOUNTS+=",$wheel_parent:$wheel_parent" ;;
esac
export MOUNTS
sha256sum -- "$WHEEL"

srun --jobid="$SLURM_JOB_ID" --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
    --container-image="$IMAGE" --container-mounts="$MOUNTS" \
    --container-workdir="$REPO" --container-writable --container-remap-root \
    bash -c 'set -e
python3 -m pip install -r "$REPO/requirements-dev.txt"
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
exec bash "$REPO/examples/disaggregated/sparse_kv_offload_repro/host_transfer_probe/run.sh"'
