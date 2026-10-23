<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek V4 Flash sparse KV host-transfer probe

This is a same-node, two-process NIXL probe for the slow 32K offload send path.
It uses the measured payload size (109,924,352 bytes) and entry count (1,284)
from the p3:d1 run in `32k-serving-2584221`. The receiver's host allocation
uses anonymous `mmap` plus `cudaHostRegister(PORTABLE | MAPPED)`, matching the
KVCM host pool's allocation style. Every payload byte is checked after each
transfer, outside the timed interval.

| Mode | NIXL write | Local step after NIXL |
| --- | --- | --- |
| `gpu` | GPU to GPU | None |
| `host` | GPU to CUDA-registered host memory | None |
| `staged` | GPU to GPU | CUDA device-to-host copies into registered host memory |

Each mode runs with one contiguous descriptor and with 1,284 descriptors
separated by 4 KiB gaps in the destination allocation. Both layouts carry the
same logical bytes. The staged case copies each descriptor to its matching
host position. It reports the NIXL phase, local copy phase, and their sum;
the sum excludes control messages and is an optimistic estimate of an
integrated staged transport. This probe does not run model serving or exercise
KVCM page ownership. The model run mixes GPU and host destinations, while each
probe case routes its full payload to one destination type. The 4 KiB gaps are
synthetic, so matching the model run's 136 ms exactly is not expected.

## Submit on the eight-GPU node

The branch changes only probe assets, so the wheel used for job `2584221`
(SHA-256 `2ee661b7e862e003dc8bfcee4bb6290e3231a573d429caa4cd0d45446f76832b`)
can be reused. From the checkout root, submit with paths visible on the
compute node:

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/devel-image.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
OUTPUT_ROOT=/absolute/path/to/results \
GPU_PAIRS="0:6" \
sbatch --export=ALL -A ACCOUNT -p PARTITION -t 02:00:00 \
    examples/disaggregated/sparse_kv_offload_repro/host_transfer_probe/run.sbatch
```

`GPU_PAIRS` names sender:receiver GPU indexes as seen inside the container;
`0:6` follows the first context GPU and the first generation GPU in the 32K
serving topology. The job requests one exclusive node. It installs
`requirements-dev.txt` and the wheel in one writable Pyxis container step.

The default matrix uses `THREADS_LIST="1 8"` to compare the production
one-thread NIXL setting with eight threads. Set `THREADS_LIST="1"` for the
smallest run. The production serving test sets `UCX_TLS=^ib` inside its
worker launcher; this probe uses the same default. Set `UCX_TLS` explicitly
to run a different protocol selection. `UCX_PROTO_INFO=y` is always enabled.

## Results

The job writes `OUTPUT_ROOT/host-transfer-JOB_ID/`:

- `summary.csv` and `summary.json`: transfer, local-copy, and summed p50/p95
  for each byte-verified case.
- `p*.sender.log` and `p*.receiver.log`: errors and UCX protocol selection.
- `p*.json`: every sample, including warmups and verification status.
- `gpu-topology.txt` and `job.log`: node topology and job setup.

If contiguous GPU-to-host remains slow, inspect its UCX protocol in the
sender log. A slow contiguous write points to the physical host path. If
the 1,284-entry case is much slower than contiguous with the same bytes,
descriptor fragmentation deserves attention. Compare `staged` total time
with `host` NIXL time before changing the full transport.
