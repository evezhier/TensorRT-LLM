<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reproduce DeepSeek V4 Flash sparse KV offload at 32K

Branch `feat/kvcm2-sparse-32k-repro` already contains the tested source changes
and this harness. Build that checkout and run the harness directly. The source
patches in `patches/` are retained for provenance; do not apply them again to
this branch.

This package reproduces the latest **post-residency-fix, unprofiled 32,768 input / 160 output token** comparison. The original experiment uses **3 prefill workers and 1 decode worker** on eight GPUs. Four-GPU serving runs use one prefill worker and one decoder. A separate matched-batch experiment checks token parity while controlling decoder occupancy. The scripts do not run Nsight Systems or search for new capacity limits.

The original run's full report is included at `../evidence/final-report.json`
and in the [benchmark report](https://chatgpt.com/space/page_bec74a2e48208191ade22eaebdb248de).

## Hardware and source prerequisites

- Serving comparison: one exclusive **8 × NVIDIA B200** node, **160 allocated CPUs and 1,024 GiB host RAM**. Three TP2/EP2 prefill workers use GPUs 0–1, 2–3 and 4–5. One TP2/EP2 decoder uses GPUs 6–7, with attention DP enabled across its two ranks. Run both modes sequentially on the same node.
- Four-GPU serving comparison: one **4 × NVIDIA GB300** node. One TP2/EP2 prefill worker uses GPUs 0–1 and one TP2/EP2 decoder uses GPUs 2–3. This is a p1:d1 result; the original p3:d1 throughput and validated capacity limits do not transfer to it.
- Matched controls: **4 × B200**, 96 CPUs and 768 GiB host RAM were used. One TP2 prefill worker and one TP2 decoder; this is a separate experiment.
- Full **43-layer DeepSeek V4 Flash** weights in `$MODEL_DIR`; a working runtime with native libraries and Python bindings. The preflight checks that `Batch.reorder` is available and that Python imports the selected checkout in source mode or the installed package in wheel mode. If `MODEL_DIR` is unset, the scripts use `$LLM_MODELS_ROOT/DeepSeek-V4-Flash`.
- Original container: `trtllm_pytorch-26.08-py3-x86_64-ubuntu24.04-skip-tritondevel-202609210610-19345.sqsh`, Python 3.12, SM100 build. The integration-test Python dependencies must be installed; the harness imports `tests/integration/defs/accuracy/test_disaggregated_serving.py`. Weights, container, native binaries and dependency directories are not bundled.

The original measured checkout was `build/deepseek-v4-sparse-disagg-e2e`, branch
`experiment/deepseek-v4-sparse-disagg-e2e`, at
**`f4c197fdea4baad546df0e6bba4065ce04e52213`** with two host-memory-sizing
files modified. This branch includes those source changes.

Initialize submodules in this checkout, then build it in the compatible
container:

```bash
git submodule update --init --recursive
python3 scripts/build_wheel.py --use_ccache -a "100-real" --skip_building_wheel --linking_install_binary
pip install -e .
```

To run the build and dependency installation in a Slurm job, submit the
companion script with the checkout path, an Enroot SquashFS path or existing
container name, GPU product, and CPU architecture:

```bash
sbatch --gres=gpu:gb300:1 --constraint=aarch64 \
    examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/build_trtllm.sbatch \
    /absolute/path/to/checked-out/TensorRT-LLM \
    /absolute/path/to/compatible-container.sqsh gb300 aarch64
```

Use your cluster's GPU type and CPU feature names for `--gres` and
`--constraint`. The script requests 64 CPUs, 256 GiB RAM, and four hours. In
the container it runs `python scripts/build_wheel.py --cuda_architectures
"native" -D FAST_BUILD=ON --use_ccache`, then `pip3 install -e .`, then
`pip install -r requirements-dev.txt`. The checkout and SquashFS must be
visible on the allocated node. The container root is writable for the Python
installs; installs into a temporary image container are local to that Slurm
job. Build artifacts written into the checkout persist.

Rebuild native `bindings` and `th_common`, including their `tensorrt_llm`
dependency. `reference/binary-manifest.json` records the original built-library
hashes. The original measurements preceded the latest documentation-only
upstream commit; this branch preserves the tested executable code.

## Workload and settings

Every request uses the **same synthetic astronomy paragraph**, repeated/truncated and followed by a fixed question. The tokenizer constructs exactly 32,768 tokens including BOS; the harness saves `prompt-tokens.json` and its hash. This is not a real dataset or a unique-prompt workload. **Prefill prefix reuse is enabled; decode block reuse is disabled.**

Generation is greedy (`temperature=0`, `seed=0`, `ignore_eos=true`, `min_tokens=max_tokens=160`). **Speculative decoding is disabled.** Prefill produces the first output token, leaving **159 tokens per request** for the decoder.

| Setting | Offload disabled (`resident`) | Offload enabled (`offload`) |
| --- | --- | --- |
| Sparse attention | DeepSeek V4 | DeepSeek V4 |
| KV dtype / tokens per block | NVFP4 / 128 | NVFP4 / 128 |
| Configured decoder max batch | 1,024 | 896 |
| Decoder free GPU memory fraction | 0.95 | 0.90 |
| Client outstanding-request limit | 1,204 | 1,696 |
| `gpu_cache_factor` | Not applicable | 2.0 |
| `host_history_size` budget | Not applicable | 256 GiB per rank |

“Disabled” means CPU sparse-KV offload is disabled; the model still uses sparse attention. The two client limits are previously validated capacities for these hardware/configuration choices. They are **not actual running decoder concurrency**, and this script does not establish that they are universal maxima. Attention DP makes the worker/rank batch setting a different quantity from total client concurrency.

Common settings: `max_seq_len=33024`, chunked prefill enabled, prefill `max_num_tokens=2048`, prefill configured max batch 128, prefill GPU fraction 0.5, `fp8_context_mla_kv_len_cap=33024`, CUDA graph padding enabled, overlap scheduling enabled, autotuning disabled. The observed serving prefill batch was 1. NIXL uses the Python transceiver, one NIXL thread, and `UCX_TLS=^ib,gdr_copy` for this single-node experiment. Fresh worker HTTP connections and retries disabled are part of the recipe.

The GB300 variant selects `UCX_TLS=^ib`, matching the faster direct host path
in the GB300 transfer probe. B200 retains `^ib,gdr_copy`. Set
`SPARSE_REPRO_UCX_TLS` to override this choice; `preflight.json` records the
effective value.

`harness/env.sh` preserves the kernel determinism controls: attention, MoE, all-reduce, top-k and MHC controls are 1, while the global `FORCE_DETERMINISTIC` is 0. Batch-dependent output differences must still be measured.

For each mode, start fresh workers and run this sequence without restarting between trials:

1. One cold request, concurrency 1.
2. One warm request, concurrency 1.
3. Warm load: 128 requests at concurrency 128.
4. Measured load: **4,096 requests**, at the mode's outstanding-request limit.
5. Stop workers gracefully to write the sparse hit/miss counters.

This gives **4,226 requests per mode**, 8,452 across both modes. Warmups are excluded from headline throughput/latency, but included in the process-wide sparse-cache counters.

## Run the scripts

Inside a prepared container with four or eight GPUs available, use a **new output directory**:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint-folder
cd /absolute/path/to/checked-out/TensorRT-LLM
# Optional: extra Python dependency directories, separated by colons.
# export EXTRA_PYTHONPATH=/absolute/python-deps:/absolute/test-deps
bash examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/reproduce_32k.sh \
    serving /absolute/results/32k-serving-01
```

The wrapper copies the measured harness into the output directory, records source/runtime provenance, samples GPU telemetry, runs resident then offload, and runs the offline analyses. It refuses to overwrite an existing output directory. It never allocates GPUs itself.
Four visible GPUs select p1:d1 and eight select p3:d1. Both serving layouts run
the same 32K prompt and report NIXL send phase, throughput, latency and parity.

To allocate the original one-node serving topology with `sbatch`, build this
checkout first and submit from the repository root:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint-folder
export REPO=/absolute/path/to/checked-out/TensorRT-LLM
export IMAGE=/absolute/compatible-container.sqsh
export OUTPUT_ROOT=/absolute/results-parent
sbatch --export=ALL --gpus=8 --partition=YOUR_B200_PARTITION --account=YOUR_ACCOUNT \
    examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch
```

The submission requests eight GPUs on one exclusive node. The script has no
explicit GPU, CPU or memory count. It writes to `32k-serving-JOB_ID` under
`OUTPUT_ROOT` and mounts the checkout, checkpoint parent and results parent
at the same paths in the container.
`REPO` must point to the checkout visible on the compute node; Slurm may
execute the batch script from its spool directory.
Set `MOUNTS` if extra Python dependencies or caches live elsewhere.

For a four-GPU GB300 node, use an Arm-compatible image and wheel built for
that node, then submit the same script with `--gpus=4`:

```bash
WHEEL=/absolute/path/to/arm-tensorrt_llm.whl \
REPO=/absolute/path/to/checked-out/TensorRT-LLM \
MODEL_DIR=/absolute/path/to/checkpoint-folder \
IMAGE=/absolute/path/to/arm-compatible-container.sqsh \
OUTPUT_ROOT=/absolute/results-parent \
sbatch --export=ALL --gpus=4 --partition=YOUR_GB300_PARTITION --account=YOUR_ACCOUNT \
    examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch
```

If a wheel built from this repro branch is available, set `WHEEL` to its
absolute path when submitting the same job:

```bash
WHEEL=/absolute/path/to/tensorrt_llm-1.4.0rc0-cp312-cp312-linux_x86_64.whl \
REPO=/absolute/path/to/checked-out/TensorRT-LLM \
MODEL_DIR=/absolute/path/to/checkpoint-folder \
IMAGE=/absolute/compatible-container.sqsh \
OUTPUT_ROOT=/absolute/results-parent \
sbatch --export=ALL --gpus=8 --partition=YOUR_B200_PARTITION --account=YOUR_ACCOUNT \
    examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch
```

The launcher installs `requirements-dev.txt`, then the wheel in a writable
container immediately before the benchmark. Wheel mode imports TensorRT-LLM
from that installation while the integration-test harness still comes from
the checkout. The wheel path and SHA-256 are recorded in `preflight.json`.
The container needs access to the package indexes used by the requirements.

### Sender-side host staging experiment

The native NIXL sender can copy GPU-backed KV fragments into a reusable pinned
host buffer before writing to the decoder's host KV pool. It defaults to
256 MiB per prefill rank on systems outside the known CPU-GPU superchip combinations.
On Arm hosts with Hopper, Blackwell, Blackwell Ultra, or Rubin GPUs, staging
defaults to off. This host-architecture and GPU-generation check is a proxy for
Grace Hopper, Grace Blackwell, and Vera Rubin systems; it does not inspect the
interconnect. Set `TRTLLM_NIXL_HOST_STAGING_MB=0` to force direct transfer or
set a positive MiB value to force staging on any system. If the buffer is full
or a transfer exceeds it, the sender uses the direct path. Only GPU-to-host KV
fragments use this buffer. The sender retains leased memory through uncertain
NIXL completion.

Build a **new wheel from this checkout** before the run. The wheel used for
job `2584221` does not contain host staging. The preflight records the override
and effective setting, and rejects an installed wheel that lacks the
implementation. For a controlled comparison, run once with
`TRTLLM_NIXL_HOST_STAGING_MB=0` and again with it set to `256`, using the same
wheel, model, and B200 node type:

```bash
TRTLLM_NIXL_HOST_STAGING_MB=256 WHEEL=/absolute/path/to/new-wheel.whl \
REPO=/absolute/path/to/checked-out/TensorRT-LLM \
MODEL_DIR=/absolute/path/to/checkpoint-folder \
IMAGE=/absolute/compatible-container.sqsh OUTPUT_ROOT=/absolute/results-parent \
sbatch --export=ALL --gpus=8 --partition=YOUR_B200_PARTITION --account=YOUR_ACCOUNT \
    examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_32k_serving.sbatch
```

Compare the offload case's NIXL send phase and end-to-end throughput in the
result reports. The send-phase timer includes the CUDA staging copy when this
path is enabled. A shorter send phase alone does not establish a serving
improvement; compare completed requests, output parity, and throughput too.

If you already have an eight-GPU Slurm allocation with enough CPU and RAM for
the model, use the lower-level wrapper from this checkout:

```bash
export MODEL_DIR=/absolute/path/to/checkpoint-folder
export IMAGE=/absolute/compatible-container.sqsh
export MOUNTS=/absolute/repo-parent:/absolute/repo-parent,/absolute/path/to:/absolute/path/to,/absolute/results-parent:/absolute/results-parent
export JOB_ID=YOUR_NEW_JOB_ID
bash examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/run_slurm.sh \
    serving /absolute/results-parent/32k-serving-01
```

Mount the checkout, weights, dependencies and output parent at the same paths
inside the container. The serving GPU pinning assumes the full node's GPUs 0–7
are allocated to you. The wrapper takes one Slurm step for the complete sequence
and leaves the allocation under your control.

For matched parity controls, use a separate four-GPU allocation/container and fresh output:

```bash
# In the four-GPU container; CUDA_VISIBLE_DEVICES lists your four allocated devices.
bash examples/disaggregated/sparse_kv_offload_repro/reproduce-32k-p3d1/reproduce_32k.sh \
    matched /absolute/results/32k-matched-01
```

This runs one prefill worker and one decoder. After cold/warm requests, it gates decode until all KV has arrived for batches **1, 96, 144 and 1,204** in both modes. Offload additionally runs **1,696 twice**. A 512-request ungated churn trial at concurrency 128 occurs between the 96 and 144 controls. The packaged controller automates that step in both modes; the original resident churn trial was manually enqueued. Each gated run requires both rank barriers to report the target count. Verify the recorded scheduled counts also remain at the target for all 159 decode iterations before calling a result a matched-batch comparison. These controls do not measure normal serving latency.

## Read the results

| Output | Purpose |
| --- | --- |
| `preflight.json`, optional `source-working-tree.patch` | Runtime versions, wheel and model hashes; source revision and local changes when Git metadata is available |
| `cases/*/config.json`, `prompt-tokens.json` | Exact resolved configuration and input token vector |
| `cases/*/*-results.json` | All output token IDs, client latency/TTFT, completions and command |
| `cases/*/*-perf.json`, `*-metrics.json` | Worker timing and iteration records |
| `cases/*/native-transfer/` | NIXL transfer timing CSVs |
| `cases/offload*/sparse-counters-rank*.json` | Sparse hits/misses written at shutdown |
| `ratio-metrics.csv`, `ratio-analysis.json` | Primary serving metrics, actual occupancy and parity |
| `prefill-feed.json` | KV-ready arrival rate from the active prefill workers |
| `fix-analysis.json` | Per-trial summaries, exact token-multiset comparisons and cache hit rate |

For serving analysis only, rerun `python3 OUTPUT/analyze_ratios.py`, then `python3 OUTPUT/analyze_prefill_feed.py`, then `python3 OUTPUT/analyze_fix.py`. Use the original output directory: raw records retain native-transfer paths. For matched runs, use `analyze_fix.py` alone. Analysis uses standard-library Python; launching workers needs the prepared runtime.

**E2E output throughput** is `4096 × 160 / client trial wall time`. **Decoder throughput** is `4096 × 159 / (last decode token time − first decode forward start)`. The decoder window excludes initial prefill/transfer/receive-wait before its first forward, but includes idle gaps and transfer-induced starvation inside that window. It is not GPU-active-time-only throughput. GPU forward event duration and wall-clock iteration cadence are reported separately.

**Actual concurrency** is the time-weighted number of requests between first decode forward and final decode token; scheduled per-iteration median/p95/peak provide a second view. **NIXL send latency** is the send task's transfer-phase elapsed time, not an isolated link-bandwidth measurement. **GPU sparse hit rate** is `sum(hits) / (sum(hits) + sum(misses))` across both decoder ranks, sampled at shutdown to avoid per-step synchronization. Prefill supply rate uses KV-transfer-start timestamps in the common central 10–90% window, before transfer.

The latest reference results are:

| Metric | Offload disabled | Offload enabled |
| --- | ---: | ---: |
| Offered client concurrency | 1,204 | 1,696 |
| Mean actual decoder concurrency | 85.80 | 96.06 |
| Scheduled decoder concurrency median / p95 / peak | 88 / 90 / 95 | 98 / 103 / 107 |
| Decoder tokens/s | 3,386.4 | 3,416.0 |
| E2E output tokens/s | 3,310.7 | 3,298.8 |
| Median E2E latency, seconds | 54.45 | 77.06 |
| Median TTFT, seconds | 50.40 | 72.66 |
| Median GPU forward, ms | 24.28 | 27.65 |
| Median NIXL send phase, ms | 5.14 | 133.47 |
| GPU sparse hit rate, including warmups | Not applicable | 87.12% |

Offload allocated approximately **6.14 GiB GPU sparse arena and 47.94 GiB CPU history per rank**; the 256 GiB host-history setting is a budget, not the physical allocation. The actual serving occupancy is far below validated capacity, so the headline serving throughput is supply-limited for this prompt/reuse configuration.

Parity compares full **token-ID vectors**, using their multisets because identical requests can complete in different orders. Request completion is a separate check. The 4,096-request serving trials had **4,082 exact multiset matches**, with the same 13 unique outputs but different frequencies: **strict serving multiset parity did not pass**. Cold and single-request warm trials matched exactly. The 128-request warm-load trial also did not match strictly. The separate gated controls matched exactly across modes at 1/96/144/1,204, and the offload 1,696 repeat matched itself. Do not convert successful HTTP completion into a parity-pass claim.

The packaged benchmark and analysis logic comes from the completed runs. The new launch wrappers were checked for syntax and configuration, and offline analysis was checked against the saved evidence; no new full-model E2E was run while preparing this handoff. Startup with fresh compilation caches and performance on a different node may differ.
