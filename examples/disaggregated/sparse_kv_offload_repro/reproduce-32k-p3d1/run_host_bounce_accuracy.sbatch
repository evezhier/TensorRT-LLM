#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-host-bounce-accuracy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
set -euo pipefail

: "${REPO:?Set REPO to the TensorRT-LLM checkout}"
: "${IMAGE:?Set IMAGE to the enroot image or container name}"
REPO=$(realpath -- "$REPO")
build_dir="$REPO/cpp/build"
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    echo "No configured C++ build at $build_dir; build the wheel from this checkout first" >&2
    exit 2
fi

container_args=(--container-image="$IMAGE")
if [[ "$IMAGE" != *.sqsh && "$IMAGE" != *.squashfs ]]; then
    container_args=(--container-name="$IMAGE")
fi
mounts="$REPO:$REPO"
git_common_dir=$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)
if [[ "$git_common_dir" != "$REPO/"* ]]; then
    mounts+=",$git_common_dir:$git_common_dir"
fi
export build_dir

srun --nodes=1 --ntasks=1 \
    "${container_args[@]}" \
    --container-mounts="$mounts" \
    --container-workdir="$REPO" \
    --container-writable --container-remap-root \
    bash -c 'set -euo pipefail
cmake --build "$build_dir" --target bounceAgentE2ETest --parallel 16
cd "$build_dir/tests/unit_tests/executor/bounce"
export CUDA_MODULE_LOADING=LAZY
if ! test_output=$(./bounceAgentE2ETest \
    --gtest_filter=BounceAgentE2E.Sparse32kPayloadIsByteExactAcrossHostRegistrations \
    --gtest_color=no 2>&1); then
    printf "%s\n" "$test_output"
    exit 1
fi
printf "%s\n" "$test_output"
if [[ "$test_output" != *"[  PASSED  ] 1 test."* ]]; then
    echo "Byte-exact test did not run to completion (possibly skipped)" >&2
    exit 1
fi'
