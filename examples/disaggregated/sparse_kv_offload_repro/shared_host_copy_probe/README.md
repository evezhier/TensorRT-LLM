<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared host CUDA copy probe

This probe measures a same-node CUDA device-to-host copy into a `memfd` mapping
shared by two processes. The sender process maps and registers the host memory
with CUDA, copies from its GPU, and synchronizes its stream. The receiver
process checks every payload byte and verifies that destination gaps were not
modified. No NIXL, UCX, model, or TensorRT-LLM wheel is required. The scatter
cases require Triton in the container.

It uses the 32K serving transfer payload (109,924,352 bytes). Three baseline
cases use `cudaMemcpyAsync`: one contiguous copy, 1,284 adjacent copies, and
1,284 copies with 4 KiB destination gaps. Two additional cases use one Triton
kernel launch to scatter the same 1,284 segments, with and without gaps, into
the CUDA-mapped host buffer. This isolates copy-call overhead from GPU scatter
cost. The timing includes sender submission and stream synchronization; JIT
compilation is excluded by warmup. Verification and receiver coordination are
outside the measured interval.

Submit on a GB300 node with the same container used for the serving benchmark:

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
OUTPUT_ROOT=/absolute/path/to/results \
sbatch --export=ALL --gpus=1 -A YOUR_ACCOUNT -p gb300 \
  examples/disaggregated/sparse_kv_offload_repro/shared_host_copy_probe/run.sbatch
```

The result is `$OUTPUT_ROOT/shared-host-copy-$SLURM_JOB_ID.json`; the Slurm log
also prints the measurements. Compare `p50_ms` by `method`, `segments`, and
`gap_bytes`, then compare with the NIXL send phase in
`32k-serving-*/ratio-metrics.csv`. These timings are diagnostic, not a serving
result: the probe does not use the cache manager's host pool, whose allocation
and lifetime would need a transport integration. A fast result establishes
that the copy mechanism is worth integrating; it does not show an end-to-end
serving speedup.
