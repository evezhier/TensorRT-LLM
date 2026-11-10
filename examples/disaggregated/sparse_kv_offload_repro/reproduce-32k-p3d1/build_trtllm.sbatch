#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=trtllm-build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=04:00:00
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: sbatch --gres=gpu:PRODUCT:1 --constraint=CPU_FEATURE $0 REPO ENROOT_SOURCE GPU_PRODUCT CPU_ARCH" >&2
    exit 2
fi

repo=$(realpath -- "$1")
enroot_source=$2
gpu_product=${3,,}
cpu_arch=$4

if [[ ! -f "$repo/scripts/build_wheel.py" || ! -f "$repo/requirements-dev.txt" ]]; then
    echo "TensorRT-LLM checkout or requirements-dev.txt not found at $repo" >&2
    exit 2
fi

case "$gpu_product" in
    h100|h200|b200|gb200|b300|gb300) ;;
    *) echo "Unsupported GPU product: $gpu_product" >&2; exit 2 ;;
esac

case "$cpu_arch" in
    x86_64|aarch64) ;;
    *) echo "Unsupported CPU architecture: $cpu_arch" >&2; exit 2 ;;
esac

if [[ $(uname -m) != "$cpu_arch" ]]; then
    echo "Allocated node has CPU architecture $(uname -m), expected $cpu_arch" >&2
    exit 2
fi

if [[ "$enroot_source" == *.sqsh || "$enroot_source" == *.squashfs ]]; then
    container_args=(--container-image="$enroot_source")
else
    container_args=(--container-name="$enroot_source")
fi

container_mounts="$repo:$repo"
git_common_dir=$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)
if [[ "$git_common_dir" != "$repo/"* ]]; then
    container_mounts+=",$git_common_dir:$git_common_dir"
fi

echo "Building $repo for $gpu_product on $cpu_arch using CUDA architecture native"
srun --nodes=1 --ntasks=1 \
    "${container_args[@]}" \
    --container-writable --container-remap-root \
    --container-mounts="$container_mounts" \
    --container-workdir="$repo" \
    bash -c 'python scripts/build_wheel.py --cuda_architectures "native" -D FAST_BUILD=ON --use_ccache && pip3 install -e . && pip install -r requirements-dev.txt'
