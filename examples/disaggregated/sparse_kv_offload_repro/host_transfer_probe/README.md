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

| Mode | NIXL write | Local CUDA step |
| --- | --- | --- |
| `gpu` | GPU to GPU | None |
| `host` | GPU to CUDA-registered host memory | None |
| `staged` | GPU to GPU | CUDA device-to-host copies into registered host memory |
| `sender_staged` | CUDA-registered host memory to host memory | CUDA device-to-host copy on sender before NIXL |

Each mode runs with one contiguous descriptor and with 1,284 descriptors
separated by 4 KiB gaps in the destination allocation. Both layouts carry the
same logical bytes. The staged case copies each descriptor to its matching
host position. The sender-staged case copies the contiguous GPU source into a
pinned sender host buffer before submitting the same descriptors to NIXL.
Both staged cases report the NIXL phase, local copy phase, and their sum;
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

The default matrix uses one NIXL thread, matching the production serving
run. It includes all four modes. Set `MODES="sender_staged"` to run only the
new sender-side route, or `MODES="host sender_staged"` for a direct comparison.
Set `THREADS_LIST="1 8"` to compare eight threads as an additional
diagnostic. A failed case is recorded in `failures.tsv`; completed cases
still appear in `summary.csv`. The production serving test sets `UCX_TLS=^ib`
inside its worker launcher; this probe uses the same default. Set `UCX_TLS` explicitly
to run a different protocol selection. `UCX_PROTO_INFO=y` is always enabled.

## Results

The job writes `OUTPUT_ROOT/host-transfer-JOB_ID/`:

- `summary.csv` and `summary.json`: transfer, local-copy, and summed p50/p95
  for each byte-verified case.
- `p*.sender.log` and `p*.receiver.log`: errors and UCX protocol selection.
- `p*.json`: every sample, including warmups and verification status.
- `failures.tsv`: case and exit code for any native abort or verification failure.
- `gpu-topology.txt` and `job.log`: node topology and job setup.

The sender-staged case needs host scratch on the sender and no receiver GPU
landing buffer. Its total excludes control messages, KVCM page ownership,
and overlap with concurrent serving work. The local copy and NIXL send are
timed separately, then summed for each sample.

If contiguous GPU-to-host remains slow, inspect its UCX protocol in the
sender log. A slow contiguous write points to the physical host path. If
the 1,284-entry case is much slower than contiguous with the same bytes,
descriptor fragmentation deserves attention. Compare `staged` total time
with `host` NIXL time before changing the full transport.

## Initial GB300 result

Job `2584536` ran GPU 0 to GPU 6 on `prenyx0210` with `UCX_TLS=^ib` and the
wheel used for the 32K serving run. Each completed case verified every byte;
the table uses eight measured samples after two warmups with one NIXL thread.

| Destination | Entries | NIXL p50, ms | Local copy p50, ms | Sum p50, ms |
| --- | ---: | ---: | ---: | ---: |
| GPU | 1 | 0.187 | 0 | 0.187 |
| GPU | 1,284 | 5.349 | 0 | 5.349 |
| Host | 1 | 220.727 | 0 | 220.727 |
| Host | 1,284 | 230.277 | 0 | 230.277 |
| Staged GPU then host | 1 | 0.162 | 2.016 | 2.178 |
| Staged GPU then host | 1,284 | 4.979 | 6.412 | 11.394 |

UCX selected `software emulation | sysv/memory` for GPU-to-host writes and
`zero-copy | cuda_ipc/cuda` for GPU-to-GPU writes. The host route is slow even
with one descriptor; fragmentation adds about 10 ms to the direct host case.
The staged sum is a measured lower bound for a future integrated path, since
it excludes control and KVCM page-lifetime work.

The optional eight-thread sweep completed the GPU cases and the contiguous
host case, then the 1,284-entry host sender aborted inside UCX with
`gdr_unpin_buffer failed. ret:22` followed by `mem type pack failed`. It
produced no result for that case. The production serving run used one NIXL
thread, and the default probe matrix now matches it.
