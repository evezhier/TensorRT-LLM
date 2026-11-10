#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-offload-e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
set -euo pipefail

: "${SLURM_JOB_ID:?Submit this script with sbatch}"
: "${REPO:?Set REPO to the TensorRT-LLM checkout}"
: "${IMAGE:?Set IMAGE to the enroot image or container name}"
: "${WHEEL:?Set WHEEL to a wheel built from this checkout}"
: "${MODEL_DIR:?Set MODEL_DIR to the DeepSeek V4 Flash checkpoint}"
: "${DATASET_DIR:?Set DATASET_DIR to the local LongBench dataset directory}"

REPO=$(realpath -- "$REPO")
WHEEL=$(realpath -- "$WHEEL")
MODEL_DIR=$(realpath -- "$MODEL_DIR")
DATASET_DIR=$(realpath -- "$DATASET_DIR")

if [[ ! -f "$WHEEL" || ! -f "$MODEL_DIR/config.json" || ! -d "$DATASET_DIR" ]]; then
    echo "Expected a wheel, model config.json, and LongBench dataset directory" >&2
    exit 2
fi

container_args=(--container-image="$IMAGE")
if [[ "$IMAGE" != *.sqsh && "$IMAGE" != *.squashfs ]]; then
    container_args=(--container-name="$IMAGE")
fi

mounts="$REPO:$REPO,$MODEL_DIR:$MODEL_DIR,$DATASET_DIR:$DATASET_DIR"
git_common_dir=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
if [[ "$git_common_dir" != "$REPO/"* ]]; then
    mounts+=",$git_common_dir:$git_common_dir"
fi
case "$WHEEL" in
    "$REPO"/*|"$MODEL_DIR"/*|"$DATASET_DIR"/*) ;;
    *)
        wheel_parent=$(dirname -- "$WHEEL")
        mounts+=",$wheel_parent:$wheel_parent"
        ;;
esac

export REPO WHEEL MODEL_DIR DATASET_DIR
srun --nodes=1 --ntasks=1 \
    "${container_args[@]}" \
    --container-mounts="$mounts" \
    --container-workdir="$REPO" \
    --container-writable --container-remap-root \
    bash -c 'set -euo pipefail
python3 -m pip install -r "$REPO/requirements-dev.txt"
python3 -m pip install --force-reinstall --no-deps "$WHEEL"
python3 -c "import torch; assert torch.cuda.device_count() >= 4"
models_root=$(mktemp -d)
mkdir -p "$models_root/datasets/Xnhyacinth"
ln -s "$MODEL_DIR" "$models_root/DeepSeek-V4-Flash"
ln -s "$DATASET_DIR" "$models_root/datasets/Xnhyacinth/LongBench"
LLM_MODELS_ROOT="$models_root" python3 -m pytest -v \
    tests/integration/defs/accuracy/test_disaggregated_serving.py::TestDeepSeekV4Flash::test_kv_cache_offload'
