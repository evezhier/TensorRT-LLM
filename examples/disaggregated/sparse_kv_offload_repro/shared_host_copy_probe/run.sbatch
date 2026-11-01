#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-shared-host-copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
set -euo pipefail

: "${REPO:?Set REPO to the benchmark checkout}"
: "${IMAGE:?Set IMAGE to the GB300 container image}"
: "${WHEEL:?Set WHEEL to the TensorRT-LLM wheel built from this checkout}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent results directory}"
REPO=$(realpath -- "$REPO")
WHEEL=$(realpath -- "$WHEEL")
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
mkdir -p -- "$OUTPUT_ROOT"
if [[ ! -f "$WHEEL" ]]; then
    echo "Wheel not found: $WHEEL" >&2
    exit 2
fi
probe="$REPO/examples/disaggregated/sparse_kv_offload_repro/shared_host_copy_probe/probe.py"
output="$OUTPUT_ROOT/shared-host-copy-$SLURM_JOB_ID.json"
mounts="$REPO:$REPO,$OUTPUT_ROOT:$OUTPUT_ROOT"
case "$WHEEL" in
    "$REPO"/*|"$OUTPUT_ROOT"/*) ;;
    *)
        wheel_parent=$(dirname -- "$WHEEL")
        mounts+=",$wheel_parent:$wheel_parent"
        ;;
esac
export WHEEL probe output

srun --ntasks=1 --container-image="$IMAGE" --container-mounts="$mounts" \
    --container-workdir="$REPO" --container-writable --container-remap-root \
    bash -c 'set -e
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
exec python3 "$probe" --kvcm --output "$output"'
echo "Result: $output"
