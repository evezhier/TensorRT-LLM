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

## Run a UCX sweep on GB300

Use an Arm-compatible image and TensorRT-LLM wheel from this checkout. A
four-GPU GB300 node defaults to sender GPU 0 and receiver GPU 2; set
`REQUIRE_GB300=1` to reject a node without an Arm host and SM 103 GPUs. This
check is a platform proxy; inspect `gpu-topology.txt` for the actual GPU pair.
From the checkout root, submit with paths visible on the compute node:

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/arm-devel-image.sqsh \
WHEEL=/absolute/path/to/arm-tensorrt_llm.whl \
OUTPUT_ROOT=/absolute/path/to/results \
REQUIRE_GB300=1 GPU_PAIRS="0:2" \
UCX_SWEEP="baseline all cuda_ipc_tcp put_zcopy" \
MODES="gpu host sender_staged" \
sbatch --export=ALL -A ACCOUNT -p PARTITION -t 02:00:00 \
    examples/disaggregated/sparse_kv_offload_repro/host_transfer_probe/run.sbatch
```

`GPU_PAIRS` names sender:receiver GPU indexes as seen inside the container;
the default is `0:2` with four visible GPUs or `0:6` with eight. The job
requests one exclusive node. It installs
`requirements-dev.txt` and the wheel in one writable Pyxis container step.
Only the probe scripts changed for the sweep, so an existing compatible wheel
can be reused.

The default sweep repeats the same byte-verified cases with four UCX settings:
`baseline` keeps the caller's `UCX_TLS` or defaults to `^ib`, matching the
32K serving run; `all` sets `UCX_TLS=all`; `cuda_ipc_tcp` allows
`cuda_ipc,cuda_copy,sm,self,tcp`; and `put_zcopy` combines `UCX_TLS=all`,
`UCX_RNDV_SCHEME=put_zcopy`, and `UCX_CUDA_IPC_ENABLE_MNNVL=n`. Set
`UCX_SWEEP=baseline` to run one setting. Each setting starts fresh sender and
receiver processes, so UCX reads its environment again. `UCX_PROTO_INFO=y`
records protocol selection in each sender log; it can be overridden.

The default matrix uses one NIXL thread, matching the production serving run.
It includes all four modes. Set `MODES="host sender_staged"` to compare direct
GPU-to-host transfer with sender staging, or include `gpu` for a GPU baseline.
Set `THREADS_LIST="1 8"` to compare eight threads as an additional
diagnostic. A failed case is recorded in `failures.tsv`; completed cases
still appear in `summary.csv`. The original eight-GPU B200 run can use
`GPU_PAIRS=0:6` and omit `REQUIRE_GB300`.

## Results

The job writes `OUTPUT_ROOT/host-transfer-JOB_ID/`:

- `summary.csv` and `summary.json`: the UCX setting, transfer, local-copy, and
  summed p50/p95 for each byte-verified case, plus speedup against the matching
  `baseline` case when it completed.
- `<setting>/p*.sender.log` and `<setting>/p*.receiver.log`: errors and UCX
  protocol selection for that setting.
- `<setting>/p*.json`: every sample, including warmups and verification status.
- `failures.tsv`: case and exit code for any native abort or verification failure.
- `gpu-topology.txt` and `job.log`: node topology and job setup.

The sender-staged case needs host scratch on the sender and no receiver GPU
landing buffer. Its total excludes control messages, KVCM page ownership,
and overlap with concurrent serving work. The local copy and NIXL send are
timed separately, then summed for each sample.

Compare rows with the same GPU pair, mode, descriptor count, and payload across
settings. Inspect the corresponding sender logs to see which UCX protocol was
selected. A faster synthetic transfer still needs confirmation with the full
32K serving run's NIXL send phase and completed-request throughput.

If contiguous GPU-to-host remains slow, inspect its UCX protocol in the
sender log. A slow contiguous write points to the physical host path. If
the 1,284-entry case is much slower than contiguous with the same bytes,
descriptor fragmentation deserves attention. Compare `staged` total time
with `host` NIXL time before changing the full transport.

## Initial B200 result

Job `2584536` ran GPU 0 to GPU 6 on a B200 node (`prenyx0210`) with
`UCX_TLS=^ib` and the wheel used for the 32K serving run. Each completed case
verified every byte. The table uses eight measured samples after two warmups
with one NIXL thread.

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

## Sender-side host staging result

Job `2584773` ran GPU 0 to GPU 6 on a B200 node (`prenyx0230`) with the same
wheel, `UCX_TLS=^ib`, and one NIXL thread. All four cases completed with every
byte verified. Medians use eight samples after two warmups.

| Mode | Entries | NIXL p50, ms | CUDA copy p50, ms | Sum p50, ms |
| --- | ---: | ---: | ---: | ---: |
| Direct GPU to host | 1 | 224.887 | 0 | 224.887 |
| Sender host staging | 1 | 15.475 | 2.020 | 17.491 |
| Direct GPU to host | 1,284 | 232.858 | 0 | 232.858 |
| Sender host staging | 1,284 | 12.640 | 2.014 | 14.654 |

For 1,284 entries, sender-side staging reduced the measured copy-plus-send
median by about 15.9 times relative to direct GPU-to-host NIXL on this node.
UCX selected `zero-copy | cma/memory` for host-to-host writes, while the direct
GPU-to-host write used `software emulation | sysv/memory`. The contiguous
sender-staged case was slower than its fragmented counterpart; the probe does
not establish why. These are synthetic transport measurements on B200, not
an end-to-end serving result or a GB300 measurement of sender-side staging.
