#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${OUTPUT:?Set OUTPUT to a fresh directory for probe results}"
if [[ -e "$OUTPUT" ]]; then
    echo "Output already exists: $OUTPUT" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$OUTPUT")"
mkdir -- "$OUTPUT"
OUTPUT=$(realpath -- "$OUTPUT")
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export REPO="${REPO:-$(realpath -- "$here/../../../..")}"
exec > >(tee "$OUTPUT/job.log") 2>&1

export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${WHEEL:-}" ]]; then
    export PYTHONPATH="${EXTRA_PYTHONPATH:-}"
else
    export PYTHONPATH="$REPO${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}"
fi
cd -- "$OUTPUT"
export UCX_TLS="${UCX_TLS:-^ib}"
export UCX_MM_ERROR_HANDLING="${UCX_MM_ERROR_HANDLING:-y}"
export UCX_PROTO_INFO="${UCX_PROTO_INFO:-y}"
UCX_SWEEP="${UCX_SWEEP:-baseline all cuda_ipc_tcp put_zcopy}"
baseline_ucx_tls=$UCX_TLS
baseline_rndv_scheme="${UCX_RNDV_SCHEME:-}"
baseline_mnnvl="${UCX_CUDA_IPC_ENABLE_MNNVL:-}"
THREADS_LIST="${THREADS_LIST:-1}"
MODES="${MODES:-gpu host staged sender_staged}"
SEGMENTS_LIST="${SEGMENTS_LIST:-1 1284}"
PAYLOAD_BYTES="${PAYLOAD_BYTES:-109924352}"
GAP_BYTES="${GAP_BYTES:-4096}"
SAMPLES="${SAMPLES:-8}"
WARMUP="${WARMUP:-2}"
TIMEOUT_MS="${TIMEOUT_MS:-30000}"

echo "Output: $OUTPUT"
echo "Host: $(hostname), job: ${SLURM_JOB_ID:-local}"
echo "UCX_SWEEP=$UCX_SWEEP, THREADS_LIST=$THREADS_LIST, MODES=$MODES"
echo "PAYLOAD_BYTES=$PAYLOAD_BYTES, SEGMENTS_LIST=$SEGMENTS_LIST, GAP_BYTES=$GAP_BYTES"
if [[ -n "${WHEEL:-}" ]]; then
    sha256sum -- "$WHEEL"
fi
python3 - <<'PY'
import os
import platform
from pathlib import Path

import tensorrt_llm
import torch

source = Path(tensorrt_llm.__file__).resolve()
repo = Path(os.environ["REPO"]).resolve()
print("TensorRT-LLM:", source)
if os.environ.get("WHEEL") and source.is_relative_to(repo):
    raise RuntimeError("Wheel mode imported TensorRT-LLM from the source checkout")
gpus = [
    (torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
    for i in range(torch.cuda.device_count())
]
print("Host architecture:", platform.machine(), "GPUs:", gpus)
if os.environ.get("REQUIRE_GB300") == "1" and (
    platform.machine().lower() != "aarch64"
    or not gpus
    or any(sm != (10, 3) for _, sm in gpus)
):
    raise RuntimeError("REQUIRE_GB300 needs an Arm host with visible SM 103 GPUs")
PY
nvidia-smi -L
gpu_count=$(nvidia-smi -L | wc -l)
if [[ -z "${GPU_PAIRS:-}" ]]; then
    case "$gpu_count" in
        4) GPU_PAIRS=0:2 ;;
        8) GPU_PAIRS=0:6 ;;
        2) GPU_PAIRS=0:1 ;;
        *) echo "Set GPU_PAIRS for $gpu_count visible GPUs" >&2; exit 2 ;;
    esac
fi
echo "GPU_PAIRS=$GPU_PAIRS"
nvidia-smi topo -m > "$OUTPUT/gpu-topology.txt"
printf 'case\trole\texit_code\n' > "$OUTPUT/failures.tsv"
failures=0

for sweep in $UCX_SWEEP; do
    case "$sweep" in
        baseline)
            export UCX_TLS="$baseline_ucx_tls"
            if [[ -n "$baseline_rndv_scheme" ]]; then
                export UCX_RNDV_SCHEME="$baseline_rndv_scheme"
            else
                unset UCX_RNDV_SCHEME
            fi
            if [[ -n "$baseline_mnnvl" ]]; then
                export UCX_CUDA_IPC_ENABLE_MNNVL="$baseline_mnnvl"
            else
                unset UCX_CUDA_IPC_ENABLE_MNNVL
            fi
            ;;
        all)
            export UCX_TLS=all
            unset UCX_RNDV_SCHEME UCX_CUDA_IPC_ENABLE_MNNVL
            ;;
        cuda_ipc_tcp)
            export UCX_TLS=cuda_ipc,cuda_copy,sm,self,tcp
            unset UCX_RNDV_SCHEME UCX_CUDA_IPC_ENABLE_MNNVL
            ;;
        put_zcopy)
            export UCX_TLS=all UCX_RNDV_SCHEME=put_zcopy UCX_CUDA_IPC_ENABLE_MNNVL=n
            ;;
        *) echo "Unknown UCX_SWEEP setting: $sweep" >&2; exit 2 ;;
    esac
    export UCX_SWEEP_NAME="$sweep"
    case_dir="$OUTPUT/$sweep"
    if [[ -e "$case_dir" ]]; then
        echo "Duplicate UCX_SWEEP setting: $sweep" >&2
        exit 2
    fi
    mkdir -- "$case_dir"
    echo "UCX sweep $sweep: TLS=$UCX_TLS RNDV=${UCX_RNDV_SCHEME:-auto} MNNVL=${UCX_CUDA_IPC_ENABLE_MNNVL:-auto}"
    for pair in $GPU_PAIRS; do
        sender_gpu=${pair%%:*}
        receiver_gpu=${pair##*:}
        if [[ ! "$pair" =~ ^[0-9]+:[0-9]+$ || "$sender_gpu" == "$receiver_gpu" ]] \
            || (( 10#$sender_gpu >= gpu_count || 10#$receiver_gpu >= gpu_count )); then
            echo "GPU_PAIRS must contain distinct numeric sender:receiver pairs: $pair" >&2
            exit 2
        fi
        for threads in $THREADS_LIST; do
            if [[ ! "$threads" =~ ^[1-9][0-9]*$ ]]; then
                echo "THREADS_LIST contains an invalid thread count: $threads" >&2
                exit 2
            fi
            for mode in $MODES; do
                case "$mode" in
                    gpu|host|staged|sender_staged) ;;
                    *) echo "MODES contains an invalid mode: $mode" >&2; exit 2 ;;
                esac
                for segments in $SEGMENTS_LIST; do
                    if [[ ! "$segments" =~ ^[1-9][0-9]*$ ]]; then
                        echo "SEGMENTS_LIST contains an invalid count: $segments" >&2
                        exit 2
                    fi
                    gap=0
                    if (( segments > 1 )); then
                        gap=$GAP_BYTES
                    fi
                    tag="p${sender_gpu}-${receiver_gpu}_t${threads}_${mode}_n${segments}"
                    port=$(python3 -c 'import socket; sock = socket.socket(); sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1]); sock.close()')
                    args=(
                        --mode "$mode" --payload-bytes "$PAYLOAD_BYTES" --segments "$segments"
                        --gap-bytes "$gap" --samples "$SAMPLES" --warmup "$WARMUP"
                        --timeout-ms "$TIMEOUT_MS" --port "$port"
                    )
                    echo "Running $tag"
                    CUDA_VISIBLE_DEVICES="$receiver_gpu" TRTLLM_NIXL_NUM_THREADS="$threads" \
                        timeout -k 10 180 python3 "$here/probe.py" --role receiver \
                        "${args[@]}" --output "$case_dir/$tag.json" \
                        > "$case_dir/$tag.receiver.log" 2>&1 &
                    receiver_pid=$!
                    sender_rc=0
                    CUDA_VISIBLE_DEVICES="$sender_gpu" TRTLLM_NIXL_NUM_THREADS="$threads" \
                        timeout -k 10 180 python3 "$here/probe.py" --role sender \
                        "${args[@]}" > "$case_dir/$tag.sender.log" 2>&1 || sender_rc=$?
                    if (( sender_rc != 0 )); then
                        kill "$receiver_pid" 2>/dev/null || true
                        wait "$receiver_pid" 2>/dev/null || true
                        echo "Sender failed for $tag (exit $sender_rc): $case_dir/$tag.sender.log" >&2
                        tail -n 30 -- "$case_dir/$tag.sender.log" >&2
                        echo "Receiver log: $case_dir/$tag.receiver.log" >&2
                        tail -n 15 -- "$case_dir/$tag.receiver.log" >&2
                        printf '%s\tsender\t%s\n' "$sweep/$tag" "$sender_rc" >> "$OUTPUT/failures.tsv"
                        failures=$((failures + 1))
                        continue
                    fi
                    receiver_rc=0
                    wait "$receiver_pid" || receiver_rc=$?
                    if (( receiver_rc != 0 )); then
                        echo "Receiver failed for $tag (exit $receiver_rc): $case_dir/$tag.receiver.log" >&2
                        tail -n 30 -- "$case_dir/$tag.receiver.log" >&2
                        printf '%s\treceiver\t%s\n' "$sweep/$tag" "$receiver_rc" >> "$OUTPUT/failures.tsv"
                        failures=$((failures + 1))
                    fi
                done
            done
        done
    done
done
python3 "$here/summarize.py" "$OUTPUT"
echo "Results: $OUTPUT/summary.csv"
if (( failures > 0 )); then
    echo "$failures probe cases failed; inspect $OUTPUT/failures.tsv and the per-case logs" >&2
    exit 1
fi
