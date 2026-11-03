#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=dsv4-sparse-accuracy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --exclusive
set -euo pipefail

: "${REPO:?Set REPO to the TensorRT-LLM checkout}"
export SPARSE_REPRO_VARIANT=matched
export SPARSE_REPRO_TRANSPORT_ACCURACY=1
bash "$REPO/examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch"
