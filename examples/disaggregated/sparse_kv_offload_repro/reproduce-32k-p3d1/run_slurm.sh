#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${REPRO_JOB_ID:?Set REPRO_JOB_ID to your current allocation}"
: "${REPRO_IMAGE:?Set REPRO_IMAGE to the compatible container image}"
: "${REPRO_MOUNTS:?Set REPRO_MOUNTS to the comma-separated container mounts}"
: "${LLM_MODELS_ROOT:?Set LLM_MODELS_ROOT}"
repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export E2E_REPO="${E2E_REPO:-$(git -C "$repro_package" rev-parse --show-toplevel)}"
case "${1:-}" in
    serving)
        repro_gpus=8
        ;;
    matched)
        repro_gpus=4
        ;;
    *)
        echo "Usage: run_slurm.sh {serving|matched} /absolute/fresh-output-directory" >&2
        exit 2
        ;;
esac
exec srun --jobid="$REPRO_JOB_ID" --ntasks=1 --cpus-per-task="${REPRO_CPUS:-160}" \
    --gpus-per-task="$repro_gpus" \
    --container-image="$REPRO_IMAGE" --container-mounts="$REPRO_MOUNTS" \
    --container-workdir="$E2E_REPO" \
    bash "$repro_package/reproduce_32k.sh" "$@"
