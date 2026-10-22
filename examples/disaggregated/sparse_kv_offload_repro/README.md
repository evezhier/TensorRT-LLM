<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reproduce DeepSeek V4 Flash sparse KV offload at 32K

The runnable package is checked in under `reproduce-32k-p3d1/`, and the
reference results are under `evidence/`. No archive extraction is needed.
Starting from branch `feat/kvcm2-sparse-32k-repro`, apply the package's
`patches/0001-preserve-sparse-residency.patch` with `git am`, then apply
`patches/0002-host-memory-sizing.patch` with `git apply` in a separate source
checkout before building. Run `reproduce-32k-p3d1/reproduce_32k.sh` from the
prepared container as described below.

This package reproduces the latest **post-residency-fix, unprofiled 32,768 input / 160 output token** comparison. The primary experiment uses **3 prefill workers and 1 decode worker**. A separate matched-batch experiment checks token parity while controlling decoder occupancy. The scripts do not run Nsight Systems or search for new capacity limits.

The original run and full report are retained at `build/sparse-disagg-e2e-artifacts/sparse-residency-fix-20261019T035901Z/final-report.json` and in the [benchmark report](https://chatgpt.com/space/page_bec74a2e48208191ade22eaebdb248de).

## Hardware and source prerequisites

- Serving comparison: one exclusive **8 × NVIDIA B200** node, **160 allocated CPUs and 1,024 GiB host RAM**. Three TP2/EP2 prefill workers use GPUs 0–1, 2–3 and 4–5. One TP2/EP2 decoder uses GPUs 6–7, with attention DP enabled across its two ranks. Run both modes sequentially on the same node.
- Matched controls: **4 × B200**, 96 CPUs and 768 GiB host RAM were used. One TP2 prefill worker and one TP2 decoder; this is a separate experiment.
- Full **43-layer DeepSeek V4 Flash** weights at `$LLM_MODELS_ROOT/DeepSeek-V4-Flash`; a working source build with native libraries and Python bindings from the same source. The preflight checks that `Batch.reorder` is available and that Python imports the selected checkout.
- Original container: `trtllm_pytorch-26.08-py3-x86_64-ubuntu24.04-skip-tritondevel-202609210610-19345.sqsh`, Python 3.12, SM100 build. The integration-test Python dependencies must be installed; the harness imports `tests/integration/defs/accuracy/test_disaggregated_serving.py`. Weights, container, native binaries and dependency directories are not bundled.

The existing shared checkout is `build/deepseek-v4-sparse-disagg-e2e`, branch `experiment/deepseek-v4-sparse-disagg-e2e`, at **`f4c197fdea4baad546df0e6bba4065ce04e52213`**. The residency fix is committed locally; two host-memory-sizing files remain modified and were also part of the tested source. Keep those modifications for exact reproduction.

For an independent checkout from this branch, apply both source patches in a
separate worktree. Set `assets` to this directory's absolute path:

```bash
assets=/absolute/path/to/examples/disaggregated/sparse_kv_offload_repro
git worktree add --detach /absolute/source-checkout feat/kvcm2-sparse-32k-repro
git -C /absolute/source-checkout am "$assets/reproduce-32k-p3d1/patches/0001-preserve-sparse-residency.patch"
git -C /absolute/source-checkout apply "$assets/reproduce-32k-p3d1/patches/0002-host-memory-sizing.patch"
git -C /absolute/source-checkout submodule update --init --recursive
```

Build/install this checkout with the repository's source-build procedure before running. Rebuild native `bindings` and `th_common`, including their `tensorrt_llm` dependency; copying Python changes onto old native binaries is insufficient. `reference/binary-manifest.json` records the original built-library hashes. The original measurements preceded the latest documentation-only upstream commit; the replayed fix preserves the tested executable code.

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

`harness/env.sh` preserves the kernel determinism controls: attention, MoE, all-reduce, top-k and MHC controls are 1, while the global `FORCE_DETERMINISTIC` is 0. Batch-dependent output differences must still be measured.

For each mode, start fresh workers and run this sequence without restarting between trials:

1. One cold request, concurrency 1.
2. One warm request, concurrency 1.
3. Warm load: 128 requests at concurrency 128.
4. Measured load: **4,096 requests**, at the mode's outstanding-request limit.
5. Stop workers gracefully to write the sparse hit/miss counters.

This gives **4,226 requests per mode**, 8,452 across both modes. Warmups are excluded from headline throughput/latency, but included in the process-wide sparse-cache counters.

## Run the scripts

Inside a prepared container with all eight GPUs available, use a **new output directory**:

```bash
export E2E_REPO=/absolute/built/TensorRT-LLM
export LLM_MODELS_ROOT=/absolute/model-root
export REPRO_ASSETS=/absolute/path/to/examples/disaggregated/sparse_kv_offload_repro
# Optional: extra Python dependency directories, separated by colons.
# export REPRO_EXTRA_PYTHONPATH=/absolute/python-deps:/absolute/test-deps
bash "$REPRO_ASSETS/reproduce-32k-p3d1/reproduce_32k.sh" serving /absolute/results/32k-serving-01
```

The wrapper copies the measured harness into the output directory, records source/runtime provenance, samples GPU telemetry, runs resident then offload, and runs the offline analyses. It refuses to overwrite an existing output directory. It never allocates GPUs itself.

For the existing shared environment, the following Slurm wrapper replaces the expired job IDs in the original launch scripts. Obtain a new exclusive eight-GPU allocation with the CPU/RAM resources above, then set its job ID:

```bash
REPRO_ROOT=/home/scratch.guijuz_coreai_1/TensorRT-LLM
REPRO_ARTIFACTS="$REPRO_ROOT/build/sparse-disagg-e2e-artifacts"
REPRO_PACKAGE="$REPRO_ARTIFACTS/reproduce-32k-p3d1"
export E2E_REPO="$REPRO_ROOT/build/deepseek-v4-sparse-disagg-e2e"
export LLM_MODELS_ROOT=/home/scratch.trt_llm_data_ci/llm-models
export PYTHONUSERBASE=/home/scratch.guijuz_coreai_1/.claude_tmp/pyuser
export REPRO_EXTRA_PYTHONPATH="$REPRO_ARTIFACTS/python-deps:$REPRO_ROOT/cpp/build/sparse-disagg.su5fQT/test-deps"
export REPRO_IMAGE=/home/scratch.guijuz_coreai_1/.claude_tmp/images/trtllm_pytorch-26.08-py3-x86_64-ubuntu24.04-skip-tritondevel-202609210610-19345.sqsh
export REPRO_MOUNTS="/home/scratch.guijuz_coreai_1:/home/scratch.guijuz_coreai_1,/home/scratch.trt_llm_data_ci:/home/scratch.trt_llm_data_ci,$REPRO_ARTIFACTS/flashinfer:/home/guijuz/.cache/tensorrt_llm/flashinfer"
export REPRO_JOB_ID=YOUR_NEW_JOB_ID
export REPRO_CPUS=160
bash "$REPRO_PACKAGE/run_slurm.sh" serving "$REPRO_ARTIFACTS/my-32k-serving-01"
```

Mount the checkout, package, weights, dependencies and output parent at the same paths inside the container. The serving GPU pinning assumes the full node's GPUs 0–7 are allocated to you. The wrapper takes one Slurm step for the complete sequence and leaves the allocation under your control.

For matched parity controls, use a separate four-GPU allocation/container and fresh output:

```bash
# In the four-GPU container; CUDA_VISIBLE_DEVICES lists your four allocated devices.
bash "$REPRO_ASSETS/reproduce-32k-p3d1/reproduce_32k.sh" matched /absolute/results/32k-matched-01
```

This runs one prefill worker and one decoder. After cold/warm requests, it gates decode until all KV has arrived for batches **1, 96, 144 and 1,204** in both modes. Offload additionally runs **1,696 twice**. A 512-request ungated churn trial at concurrency 128 occurs between the 96 and 144 controls. The packaged controller automates that step in both modes; the original resident churn trial was manually enqueued. Each gated run requires both rank barriers to report the target count. Verify the recorded scheduled counts also remain at the target for all 159 decode iterations before calling a result a matched-batch comparison. These controls do not measure normal serving latency.

## Read the results

| Output | Purpose |
| --- | --- |
| `preflight.json`, `source-working-tree.patch` | Checkout, runtime versions, model config hash and local changes |
| `cases/*/config.json`, `prompt-tokens.json` | Exact resolved configuration and input token vector |
| `cases/*/*-results.json` | All output token IDs, client latency/TTFT, completions and command |
| `cases/*/*-perf.json`, `*-metrics.json` | Worker timing and iteration records |
| `cases/*/native-transfer/` | NIXL transfer timing CSVs |
| `cases/offload*/sparse-counters-rank*.json` | Sparse hits/misses written at shutdown |
| `ratio-metrics.csv`, `ratio-analysis.json` | Primary serving metrics, actual occupancy and parity |
| `prefill-feed.json` | KV-ready arrival rate from all three prefill workers |
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
