#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-shared-host-copy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
set -euo pipefail

: "${REPO:?Set REPO to the benchmark checkout}"
: "${IMAGE:?Set IMAGE to the GB300 container image}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a persistent results directory}"
REPO=$(realpath -- "$REPO")
OUTPUT_ROOT=$(realpath -m -- "$OUTPUT_ROOT")
mkdir -p -- "$OUTPUT_ROOT"
probe="$REPO/examples/disaggregated/sparse_kv_offload_repro/shared_host_copy_probe/probe.py"
output="$OUTPUT_ROOT/shared-host-copy-$SLURM_JOB_ID.json"

srun --ntasks=1 --container-image="$IMAGE" \
    --container-mounts="$REPO:$REPO,$OUTPUT_ROOT:$OUTPUT_ROOT" \
    --container-workdir="$REPO" \
    python3 "$probe" --output "$output"
echo "Result: $output"
