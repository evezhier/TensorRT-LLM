<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# DeepSeek V4 Flash sparse KV offload transfer benchmark

This harness runs the full 43-layer DeepSeek V4 Flash checkpoint with a
32,768-token prompt and 160 output tokens. It starts a resident decode worker,
then a sparse KV offload decode worker, using the same synthetic prompt and
request sequence. The measured trial sends 4,096 requests per mode. It reports
NIXL send-phase p50 in `ratio-metrics.csv`, along with decode throughput,
latency, GPU forward time, and token parity.

This is the serving workload from `feat/kvcm2-sparse-32k-repro`, adapted to
`sparse-offload-full`. The current branch's sparse `Batch.move` API and host
pool sizing differ from the source branch. No historical performance number is
assumed for this implementation.

## Topology

Use one node with either eight visible GPUs (three TP2 prefill workers and one
TP2 decoder, p3:d1) or four visible GPUs (one TP2 prefill worker and one TP2
decoder, p1:d1). The four-GPU layout is suitable for a GB300 node, but its
throughput is not directly comparable with the eight-GPU result. Both modes
run sequentially on the same node. The script requests no CPU or memory count;
pass the GPU count to `sbatch`.

## Submit

Build a TensorRT-LLM wheel from this checkout for the target architecture and
make it available on the compute node. The container must support that GPU
and CPU architecture and allow package installation. Set the model path to
the directory containing the checkpoint's `config.json`.

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash \
OUTPUT_ROOT=/absolute/path/to/results \
sbatch --export=ALL --gpus=4 -A YOUR_ACCOUNT -p YOUR_GB300_PARTITION \
  examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch
```

Use `--gpus=8` on an eight-GPU node. Set `MOUNTS` if the default mounts for
the repository, checkpoint parent, and output root are insufficient. In wheel
mode, the job installs `requirements-dev.txt` and the selected wheel in a
writable container. Without `WHEEL`, the container must already have compatible
native libraries, and Python runs from this checkout.

The output directory is `$OUTPUT_ROOT/32k-serving-$SLURM_JOB_ID`. Every run
requires a fresh output directory. Preflight records the source revision,
wheel hash, checkpoint hash, GPU names, and effective UCX setting. It imports
the DeepSeek V4 cache manager, checks `Batch.move`, and requires the installed
wheel's cache-manager source to match this checkout.

On GB300, the harness defaults to `UCX_TLS=^ib`; elsewhere it uses
`UCX_TLS=^ib,gdr_copy`. Set `SPARSE_REPRO_UCX_TLS` to override this choice.

## Read the result

The main comparison is in `ratio-metrics.csv`: select the
`100-performance-1` rows for `resident` and `offload`. The
`p50_send_transfer_ms` column is the NIXL sender's transfer phase, excluding
its preparation and queue phases. `ratio-analysis.json` contains the full
transfer distributions and payload sizes; `fix-analysis.json` contains token
parity and sparse hit/miss counters. Raw worker timing CSVs are in each case's
`native-transfer/` directory.

The sender timer is task elapsed time, not hardware link latency. Compare
completed requests, transfer payloads, throughput, and parity before drawing
a performance conclusion. The files in `reference/` document the earlier
source-branch experiment and are not expected values for this branch.
