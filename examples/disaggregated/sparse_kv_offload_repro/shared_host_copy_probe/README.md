<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared host CUDA copy probe

This probe measures a same-node CUDA device-to-host copy into a `memfd` mapping
shared by two processes. The sender process maps and registers the host memory
with CUDA, copies from its GPU, and synchronizes its stream. The receiver
process checks every payload byte and verifies that destination gaps were not
modified. The final two cases copy the scattered data into pages allocated by
a real KVCM2 sparse host-backed pool, using CPU memcpy or a receiver-side GPU
scatter, and verify each page. They measure the complete sender-to-receiver
handoff, including receiver coordination and the host-pool copy. The probe
does not use NIXL, UCX, or model weights. It requires Triton in the container
and a TensorRT-LLM wheel built from this checkout.

It uses the 32K serving transfer payload (109,924,352 bytes). Three baseline
cases use `cudaMemcpyAsync`: one contiguous copy, 1,284 adjacent copies, and
1,284 copies with 4 KiB destination gaps. Two additional cases use one Triton
kernel launch to scatter the same 1,284 segments, with and without gaps, into
the CUDA-mapped host buffer. The sixth and seventh cases then copy from that
buffer to real KVCM2 host-backed pages by CPU and GPU, respectively. This
isolates copy-call overhead from GPU scatter cost and compares two ways to
place data in the pool. `p50_ms` covers sender submission and stream
synchronization; `handoff_p50_ms` covers sender/receiver coordination and the
host-pool copy. JIT compilation is excluded by warmup. Verification is outside
the measured intervals.

Submit on a GB300 node with the same container used for the serving benchmark:

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
OUTPUT_ROOT=/absolute/path/to/results \
sbatch --export=ALL --gpus=1 -A YOUR_ACCOUNT -p gb300 \
  examples/disaggregated/sparse_kv_offload_repro/shared_host_copy_probe/run.sbatch
```

The result is `$OUTPUT_ROOT/shared-host-copy-$SLURM_JOB_ID.json`; the Slurm log
also prints the measurements. Compare `p50_ms` by `method`, `segments`, and
`gap_bytes`; the KVCM2 cases add `handoff_p50_ms`,
`host_pool_copy_p50_ms`, and `host_pool_copy_method`. Compare with the NIXL send phase in
`32k-serving-*/ratio-metrics.csv`. This still is not a live serving result:
the probe uses a synthetic KVCM2 sparse pool and does not change serving's
transport protocol or request ownership. A favorable complete handoff result
supports implementing that protocol next.
