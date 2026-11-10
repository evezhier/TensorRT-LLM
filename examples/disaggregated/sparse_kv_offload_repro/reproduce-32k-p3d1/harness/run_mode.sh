#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
repro_harness=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$repro_harness/env.sh"
exec python3 "$repro_harness/benchmark_server.py" "$@"
