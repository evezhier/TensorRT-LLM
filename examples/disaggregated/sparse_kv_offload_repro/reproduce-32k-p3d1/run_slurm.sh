#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${REPRO_JOB_ID:?Set REPRO_JOB_ID to your current allocation}"
: "${REPRO_IMAGE:?Set REPRO_IMAGE to the compatible container image}"
: "${REPRO_MOUNTS:?Set REPRO_MOUNTS to the comma-separated container mounts}"
if [[ -z "${REPRO_MODEL_DIR:-}" && -z "${LLM_MODELS_ROOT:-}" ]]; then
    echo "Set REPRO_MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent" >&2
    exit 2
fi
repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export E2E_REPO="${E2E_REPO:-$(git -C "$repro_package" rev-parse --show-toplevel)}"
case "${1:-}" in
    serving|matched)
        ;;
    *)
        echo "Usage: run_slurm.sh {serving|matched} /absolute/fresh-output-directory" >&2
        exit 2
        ;;
esac
srun_args=(
    --jobid="$REPRO_JOB_ID" --ntasks=1 --cpus-per-task="${REPRO_CPUS:-160}"
    --container-image="$REPRO_IMAGE" --container-mounts="$REPRO_MOUNTS"
    --container-workdir="$E2E_REPO"
)
if [[ -n "${REPRO_WHEEL:-}" ]]; then
    if [[ ! -f "$REPRO_WHEEL" ]]; then
        echo "Prebuilt wheel not found: $REPRO_WHEEL" >&2
        exit 2
    fi
    exec srun "${srun_args[@]}" --container-writable --container-remap-root \
        bash -c 'set -e; python3 -m pip install --force-reinstall --no-deps "$REPRO_WHEEL"; exec bash "$@"' \
        _ "$repro_package/reproduce_32k.sh" "$@"
fi
exec srun "${srun_args[@]}" bash "$repro_package/reproduce_32k.sh" "$@"
