#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${JOB_ID:?Set JOB_ID to your current allocation}"
: "${IMAGE:?Set IMAGE to the compatible container image}"
: "${MOUNTS:?Set MOUNTS to the comma-separated container mounts}"
if [[ -z "${MODEL_DIR:-}" && -z "${LLM_MODELS_ROOT:-}" ]]; then
    echo "Set MODEL_DIR to the checkpoint folder or LLM_MODELS_ROOT to its parent" >&2
    exit 2
fi
repro_package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export REPO="${REPO:-$(realpath -- "$repro_package/../../../..")}"
case "${1:-}" in
    serving|matched)
        ;;
    *)
        echo "Usage: run_slurm.sh {serving|matched} /absolute/fresh-output-directory" >&2
        exit 2
        ;;
esac
srun_args=(
    --jobid="$JOB_ID" --ntasks=1 --cpus-per-task="${CPUS:-160}"
    --container-image="$IMAGE" --container-mounts="$MOUNTS"
    --container-workdir="$REPO"
)
if [[ -n "${WHEEL:-}" ]]; then
    if [[ ! -f "$WHEEL" ]]; then
        echo "Prebuilt wheel not found: $WHEEL" >&2
        exit 2
    fi
    exec srun "${srun_args[@]}" --container-writable --container-remap-root \
        bash -c 'set -e
python3 -m pip install -r "$REPO/requirements-dev.txt"
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
exec bash "$@"' \
        _ "$repro_package/reproduce_32k.sh" "$@"
fi
exec srun "${srun_args[@]}" bash "$repro_package/reproduce_32k.sh" "$@"
