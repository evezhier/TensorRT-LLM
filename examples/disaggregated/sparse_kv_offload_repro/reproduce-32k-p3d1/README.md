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
wheel's cache-manager and KV transfer sources to match this checkout.

On GB300, the harness defaults to `UCX_TLS=^ib`; elsewhere it uses
`UCX_TLS=^ib,gdr_copy`. Set `SPARSE_REPRO_UCX_TLS` to override this choice.

To measure the coalesced GPU transfer followed by a GPU scatter into the
offloaded KVCM2 host pool, add `SPARSE_REPRO_KV_BOUNCE_MB=256` before `sbatch`.
This enables the existing Python KV bounce transport on the prefill and
offload decode workers; the resident comparison keeps its original transport.
Each worker allocates two GPU bounce regions of this size. The sender gathers
device fragments, NIXL writes one device region, and the receiver scatters
into CUDA-registered host pages before declaring transfer complete. Check the
worker logs for `[kv-bounce] coalesced` and
`[kv-bounce] GPU scatter into CUDA-mapped host pool` to confirm the path taken.
A transfer that cannot reserve bounce space falls
back to the direct per-fragment path.
For TP2 fan-in with unequal layer-group sizes, the receiver reserves one
full-payload slot per sender and checks each sender's actual byte count before
submission. The 32K payload is about 110 MB, so one such request uses about
220 MB of the 256 MiB receive region. More overlapping requests may fall back
unless the bounce capacity is increased.

To test the credit-scheduled C++ agent bounce, set
`SPARSE_REPRO_AGENT_BOUNCE_MB=256` instead. This requires a wheel rebuilt from
this checkout. The agent gathers GPU fragments into bounded GPU staging,
transfers each granted chunk GPU to GPU, then scatters into the CUDA-mapped
host pool. Do not set `SPARSE_REPRO_KV_BOUNCE_MB` in the same run. The C++ agent
bounce gate is relaxed for this benchmark's roughly 86 KiB descriptors. The C++ agent
bounce negotiates its wire version with the peer and falls back to standard
NIXL when the peer has an incompatible version.

Before accepting agent-bounce performance, build and run the byte-exact GPU-to-host
test on the target GPU:

```bash
cmake --build cpp/build --target bounceAgentE2ETest
cpp/build/tests/unit_tests/executor/bounce/bounceAgentE2ETest \
  --gtest_filter=BounceAgentE2E.Sparse32kPayloadIsByteExactAcrossHostRegistrations
```

On GB300, submit the same check as a Slurm job after building the wheel:

```bash
REPO=/path/to/TensorRT-LLM IMAGE=/path/to/devel.sqsh \
  sbatch --export=ALL -A coreai_devtech_all -p gb300 --gpus=1 \
  examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_host_bounce_accuracy.sbatch
```

`Sparse32kPayloadIsByteExactAcrossHostRegistrations` uses the 32K serving
transfer's 1,284-descriptor, roughly 105 MiB shape and separate 32 MiB host
registrations. The serving comparison's token parity can vary between two
resident runs, so it does not by itself establish byte-copy correctness.
The Slurm script requires Google Test to report one passed test; a skipped
test fails the job.

The DeepSeek V4 Flash disaggregated offload GSM8K test in
`tests/integration/defs/accuracy/test_disaggregated_serving.py` has direct and
agent-bounce variants. The agent variant requires a log entry confirming that
a host-memory destination took the C++ bounce path. Run both against the same
checkpoint before treating a transport speedup as usable. The 4K GSM8K workload
checks model accuracy; the byte-exact test above checks the 32K payload shape.
Set `LLM_MODELS_ROOT` to the parent of `DeepSeek-V4-Flash` when running the
integration test. The direct and `host_bounce` parameter variants are scheduled
in the B200 test list and can also be run on GB300 with four GPUs.

To run both variants in one Slurm job, provide the checkpoint directory and the
local GSM8K dataset directory separately. `DATASET_DIR` must contain
`main/test-00000-of-00001.parquet`. The job mounts both directories, makes the
`LLM_MODELS_ROOT` layout expected by the test inside the container, and installs
a wheel built from this checkout. Request the four GPUs at submission time:

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash \
DATASET_DIR=/absolute/path/to/datasets/openai/gsm8k \
sbatch --export=ALL --gpus=4 -A YOUR_ACCOUNT -p YOUR_GB300_PARTITION \
  examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_deepseek_offload_e2e.sbatch
```

The Slurm log contains the GSM8K score and the result of each parameter
variant. A full run can take more than an hour because the servers start
separately for each variant.

## Compare direct and agent transfer at 32K

This job runs two offloaded cases sequentially on the same allocated GPUs. Both
use the same 32K prompt, 160 output tokens, and 4,096-request performance trial.
Only the transfer selection differs: direct NIXL versus the 256 MiB C++ agent
bounce. It requires a wheel built from this checkout, including the native
`bounce_host_submit_count` binding.

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash \
OUTPUT_ROOT=/absolute/path/to/results \
sbatch --export=ALL --gpus=4 -A YOUR_ACCOUNT -p YOUR_GB300_PARTITION \
  examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_transfer_compare.sbatch
```

Read `$OUTPUT_ROOT/32k-compare-$SLURM_JOB_ID/transport-performance.csv` for
the direct and agent send-phase p50/p95, receive time, throughput, and latency.
`transport-performance.json` also records the source, wheel, payload comparison,
and host-destination bounce count. The job fails when requests or timing records
are incomplete, payload bytes differ, or host bounce was not exercised.

For a bounded stall diagnostic, set `SPARSE_REPRO_PERF_CONCURRENCY=512` and
`SPARSE_REPRO_PERF_REQUESTS=1024` on the same submission command. Increase
concurrency on a later run to find the point where requests stop finishing.
`SPARSE_REPRO_GEN_HOST_CACHE_GIB=32` gives the generation KV cache a separate
host tier for an eviction test; the default is zero. These settings keep the
32K prompt and 160 output tokens but change the offered load, so compare only
runs with the same settings. The case log at
`cases/offload-direct-p1-d1.log` prints `SWEEP_PROGRESS` every 30 seconds with
started, response-header, first-token, completed, and failed request counts.
The benchmark passes its selected `UCX_TLS` setting through to the workers;
check their `BindingsNixlTransferAgent init` lines to verify it.

## Check model output with offload held constant

The serving `resident`/`offload` parity result changes cache residency and
batching as well as transport. To isolate the transport, run the matched-batch
accuracy job. It launches two direct-transfer offload controls and one agent
bounce offload case, all with the same checkpoint, prompt, and serving config.
It compares exact output-token multisets for the cold, warm, batch-1, and
batch-96 trials. A mismatch between the two direct controls makes the result
inconclusive; a mismatch between stable direct controls and agent bounce fails
the job. The job also requires a positive C++ bounce submission count.
Use the reference-based GSM8K test above as the model accuracy gate; exact
token equality here is only a diagnostic when the direct controls are stable.

```bash
REPO=/absolute/path/to/TensorRT-LLM \
IMAGE=/absolute/path/to/container.sqsh \
WHEEL=/absolute/path/to/tensorrt_llm.whl \
MODEL_DIR=/absolute/path/to/DeepSeek-V4-Flash \
OUTPUT_ROOT=/absolute/path/to/results \
sbatch --export=ALL --gpus=4 -A YOUR_ACCOUNT -p YOUR_GB300_PARTITION \
  examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_transport_accuracy.sbatch
```

Read `$OUTPUT_ROOT/32k-matched-$SLURM_JOB_ID/transport-accuracy.json`. A pass
establishes transport equivalence for these controlled requests; expand the
matched batches before making a broader accuracy claim.

To reanalyze a completed run without launching servers or changing its files:

```bash
python3 examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/harness/analyze_transport_accuracy.py \
  /absolute/path/to/32k-matched-JOB_ID
```

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
