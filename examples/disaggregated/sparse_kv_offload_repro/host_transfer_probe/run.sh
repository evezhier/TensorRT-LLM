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
export UCX_PROTO_INFO=y
GPU_PAIRS="${GPU_PAIRS:-0:6}"
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
echo "UCX_TLS=$UCX_TLS, GPU_PAIRS=$GPU_PAIRS, THREADS_LIST=$THREADS_LIST"
echo "MODES=$MODES"
echo "PAYLOAD_BYTES=$PAYLOAD_BYTES, SEGMENTS_LIST=$SEGMENTS_LIST, GAP_BYTES=$GAP_BYTES"
if [[ -n "${WHEEL:-}" ]]; then
    sha256sum -- "$WHEEL"
fi
python3 - <<'PY'
import os
from pathlib import Path

import tensorrt_llm

source = Path(tensorrt_llm.__file__).resolve()
repo = Path(os.environ["REPO"]).resolve()
print("TensorRT-LLM:", source)
if os.environ.get("WHEEL") and source.is_relative_to(repo):
    raise RuntimeError("Wheel mode imported TensorRT-LLM from the source checkout")
PY
nvidia-smi -L
nvidia-smi topo -m > "$OUTPUT/gpu-topology.txt"
printf 'case\trole\texit_code\n' > "$OUTPUT/failures.tsv"
failures=0

for pair in $GPU_PAIRS; do
    sender_gpu=${pair%%:*}
    receiver_gpu=${pair##*:}
    if [[ ! "$pair" =~ ^[0-9]+:[0-9]+$ || "$sender_gpu" == "$receiver_gpu" ]]; then
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
                    "${args[@]}" --output "$OUTPUT/$tag.json" \
                    > "$OUTPUT/$tag.receiver.log" 2>&1 &
                receiver_pid=$!
                sender_rc=0
                CUDA_VISIBLE_DEVICES="$sender_gpu" TRTLLM_NIXL_NUM_THREADS="$threads" \
                    timeout -k 10 180 python3 "$here/probe.py" --role sender \
                    "${args[@]}" > "$OUTPUT/$tag.sender.log" 2>&1 || sender_rc=$?
                if (( sender_rc != 0 )); then
                    kill "$receiver_pid" 2>/dev/null || true
                    wait "$receiver_pid" 2>/dev/null || true
                    echo "Sender failed for $tag (exit $sender_rc); inspect its logs" >&2
                    printf '%s\tsender\t%s\n' "$tag" "$sender_rc" >> "$OUTPUT/failures.tsv"
                    failures=$((failures + 1))
                    continue
                fi
                receiver_rc=0
                wait "$receiver_pid" || receiver_rc=$?
                if (( receiver_rc != 0 )); then
                    echo "Receiver failed for $tag (exit $receiver_rc); inspect its logs" >&2
                    printf '%s\treceiver\t%s\n' "$tag" "$receiver_rc" >> "$OUTPUT/failures.tsv"
                    failures=$((failures + 1))
                fi
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
