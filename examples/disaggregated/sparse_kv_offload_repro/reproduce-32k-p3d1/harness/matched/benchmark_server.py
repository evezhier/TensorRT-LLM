# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure disaggregated capacity, exact tokens, and streamed serving performance."""

import argparse
import asyncio
import copy
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import requests
from metrics_collection import (completed_requests, distribution, finish_metrics,
                                summarize_metrics, worker_metrics)
from defs.accuracy import test_disaggregated_serving as disagg_test
from defs.accuracy.test_disaggregated_serving import launch_disaggregated_llm

from tensorrt_llm.llmapi.tokenizer import load_hf_tokenizer


def write_json(path: Path, value: object) -> None:
    """Atomically publish one complete result."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def complete_batch(
    url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    concurrency: int,
    count: int,
    timeout_seconds: float,
) -> list[dict]:
    """Run a bounded asynchronous client, retaining token identities and timestamps."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "detokenize": False,
            "stream": True,
        }
    ).encode()
    timeout = aiohttp.ClientTimeout(total=timeout_seconds)
    connector = aiohttp.TCPConnector(limit=0)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:

        async def one(index: int) -> dict:
            async with semaphore:
                started = time.monotonic()
                tokens = []
                arrivals = []
                perf_headers = {}
                response_id = None
                try:
                    async with session.post(
                        f"{url}/v1/completions",
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "X-TRTLLM-return-metrics": "1",
                        },
                    ) as response:
                        response.raise_for_status()
                        async for line in response.content:
                            if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                                continue
                            chunk = json.loads(line[6:])
                            if "error" in chunk:
                                raise RuntimeError(str(chunk["error"]))
                            if "choices" not in chunk:
                                perf_headers.update(chunk)
                                continue
                            response_id = chunk.get("id", response_id)
                            new_tokens = [
                                token
                                for choice in chunk["choices"]
                                for token in (choice.get("token_ids") or [])
                            ]
                            if new_tokens:
                                now = time.monotonic() - started
                                tokens.extend(new_tokens)
                                arrivals.extend([now] * len(new_tokens))
                    if len(tokens) != output_tokens:
                        raise RuntimeError(
                            f"Expected {output_tokens} tokens, received {len(tokens)}"
                        )
                    return {
                        "index": index,
                        "id": response_id,
                        "token_ids": tokens,
                        "passed": True,
                        "token_arrival_seconds": arrivals,
                        "ttft_seconds": arrivals[0],
                        "latency_seconds": arrivals[-1],
                        "stream_closed_seconds": time.monotonic() - started,
                        "tpot_seconds": (arrivals[-1] - arrivals[0]) / max(1, len(tokens) - 1),
                        "perf_headers": perf_headers,
                    }
                except (
                    aiohttp.ClientError,
                    TimeoutError,
                    RuntimeError,
                    ValueError,
                    KeyError,
                ) as error:
                    return {
                        "index": index,
                        "id": response_id,
                        "token_ids": tokens,
                        "passed": False,
                        "token_arrival_seconds": arrivals,
                        "seconds": time.monotonic() - started,
                        "error": f"{type(error).__name__}: {error}",
                    }

        return await asyncio.gather(*(one(index) for index in range(count)))


def perf_offsets(directory: Path) -> dict[str, int]:
    """Remember flushed JSONL positions before one measured trial."""
    return {str(path): path.stat().st_size for path in directory.glob("*.jsonl")}


def read_new_perf(directory: Path, offsets: dict[str, int]) -> list[dict]:
    """Read only complete per-request performance records from this trial."""
    records = []
    for path in sorted(directory.glob("*.jsonl")):
        with path.open() as stream:
            stream.seek(offsets.get(str(path), 0))
            for line in stream:
                if line.endswith("\n"):
                    records.append({"source_file": path.name, **json.loads(line)})
    return records


def control_profile(workers: dict, action: str, directory: Path) -> list[dict]:
    """Use the serving API for a separate short GPU-kernel diagnostic."""
    receipts = []
    for worker in workers["generation_servers"]:
        try:
            body = {"output_dir": str(directory), "activities": ["GPU"]} if action == "start" else None
            reply = requests.post(
                f"http://{worker['host']}:{worker['port']}/{action}_profile", json=body, timeout=90
            )
            reply.raise_for_status()
            receipts.append({"success": True, "reply": reply.json()})
        except (requests.RequestException, ValueError) as error:
            receipts.append({"success": False, "error": str(error)})
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("resident", "offload"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=160)
    parser.add_argument("--host-gib", type=float, default=128)
    parser.add_argument("--gpu-fraction", type=float, default=0.95)
    parser.add_argument("--context-gpu-fraction", type=float, default=0.5)
    parser.add_argument("--context-max-batch-size", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=7200)
    parser.add_argument("--capacity-gate", action="store_true")
    parser.add_argument("--context-instances", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--context-kv-length-cap", type=int)
    args = parser.parse_args()
    allocation_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    assert len(allocation_devices) == 4, allocation_devices
    args.output.mkdir(parents=True, exist_ok=True)
    commands = args.output / "commands"
    commands.mkdir(exist_ok=True)
    perf_directory = args.output / "perf"
    perf_directory.mkdir(exist_ok=True)
    model = Path(os.environ["MODEL_DIR"])
    checkpoint = json.loads((model / "config.json").read_text())
    assert checkpoint["num_hidden_layers"] == 43
    assert args.input_tokens + args.output_tokens <= checkpoint["max_position_embeddings"]
    tokenizer = load_hf_tokenizer(str(model), trust_remote_code=True)
    paragraph = tokenizer.encode(
        "The observatory keeps a notebook of its observations. On Monday the sky was clear, "
        "on Tuesday clouds covered the valley, and on Wednesday the stars returned. "
        "The telescope uses a mirror to collect light from distant objects. ",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nQuestion: Explain how a telescope collects light and why a clear sky helps astronomers.\nAnswer:",
        add_special_tokens=False,
    )
    body_length = args.input_tokens - len(suffix) - 1
    prompt = (
        [checkpoint["bos_token_id"]]
        + (paragraph * (body_length // len(paragraph) + 1))[:body_length]
        + suffix
    )
    assert len(prompt) == args.input_tokens
    max_seq_len = ((args.input_tokens + args.output_tokens + 127) // 128) * 128
    graph_sizes = [
        size
        for size in (
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            768,
            1024,
            1280,
            1536,
            1792,
            2048,
            3072,
            4096,
        )
        if size <= args.max_batch_size
    ]
    if args.max_batch_size not in graph_sizes:
        graph_sizes.append(args.max_batch_size)
    common = {
        "tensor_parallel_size": 2,
        "moe_expert_parallel_size": 2,
        "moe_config": {"backend": "TRTLLM"},
        "disable_overlap_scheduler": False,
        "enable_chunked_prefill": True,
        "cuda_graph_config": {"batch_sizes": graph_sizes, "enable_padding": True},
        "enable_autotuner": False,
        "enable_iter_perf_stats": True,
        "iter_stats_max_iterations": -1,
        "max_batch_size": args.max_batch_size,
        "max_seq_len": max_seq_len,
        "max_num_tokens": max(2048, args.max_batch_size),
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(perf_directory),
        "kv_cache_config": {
            "dtype": "nvfp4",
            "tokens_per_block": 128,
            "enable_block_reuse": True,
            "free_gpu_memory_fraction": args.gpu_fraction,
            "host_cache_size": 0,
            "max_util_for_resume": 1.0,
            "fp8_context_mla_kv_len_cap": max_seq_len,
        },
        "cache_transceiver_config": {
            "backend": "NIXL",
            "transceiver_runtime": "PYTHON",
            "max_tokens_in_buffer": max_seq_len,
            "kv_transfer_timeout_ms": 3600000,
        },
        "sparse_attention_config": {
            "algorithm": "deepseek_v4",
            "compress_ratios": checkpoint["compress_ratios"],
            "index_head_dim": checkpoint["index_head_dim"],
            "index_topk": checkpoint["index_topk"],
            "window_size": checkpoint["sliding_window"],
        },
    }
    ctx = copy.deepcopy(common)
    gen = copy.deepcopy(common)
    ctx["max_batch_size"] = args.context_max_batch_size
    ctx["max_num_tokens"] = 2048
    ctx["cuda_graph_config"]["batch_sizes"] = [1]
    ctx["kv_cache_config"]["free_gpu_memory_fraction"] = args.context_gpu_fraction
    if args.context_kv_length_cap is not None:
        assert args.context_kv_length_cap >= max_seq_len
        ctx["kv_cache_config"]["fp8_context_mla_kv_len_cap"] = args.context_kv_length_cap
    gen["enable_attention_dp"] = True
    gen["kv_cache_config"]["enable_block_reuse"] = False
    gen["kv_cache_config"]["avg_seq_len"] = args.input_tokens + 2
    if args.mode == "offload":
        gen["sparse_attention_config"].update(
            {
                "enable_kv_cache_offload": True,
                "kv_cache_offload_config": {
                    "gpu_cache_factor": 2.0,
                    "host_history_size": int(args.host_gib * 1024**3),
                },
            }
        )
    disagg = {
        "hostname": "localhost",
        "backend": "pytorch",
        "context_servers": {"num_instances": args.context_instances},
        "generation_servers": {"num_instances": 1},
        "schedule_style": "generation_first",
        "return_perf_metrics": True,
        "perf_metrics_output_dir": str(perf_directory),
    }
    write_json(
        args.output / "config.json",
        {
            "mode": args.mode,
            "model": str(model),
            "model_layers": checkpoint["num_hidden_layers"],
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
            "prompt_kind": "Synthetic identical repeated astronomy paragraph plus fixed question",
            "context_config": ctx,
            "generation_config": gen,
            "disaggregation_config": disagg,
            "capacity_gate_installed": args.capacity_gate,
            "physical_generation_gpu_ids": allocation_devices[2:4],
            "physical_context_gpu_ids": allocation_devices[:2],
            "worker_http_connection_policy": "fresh_connection_per_request",
            "worker_http_retries_disabled": True,
            "deterministic_kernel_controls": {
                name: os.getenv(name, "0")
                for name in (
                    "FORCE_DETERMINISTIC",
                    "FORCE_ATTENTION_KERNEL_DETERMINISTIC",
                    "FORCE_MOE_KERNEL_DETERMINISTIC",
                    "FORCE_ALL_REDUCE_DETERMINISTIC",
                    "FORCE_TOPK_KERNEL_DETERMINISTIC",
                    "FORCE_MHC_KERNEL_DETERMINISTIC",
                )
            },
        },
    )
    write_json(args.output / "prompt-tokens.json", prompt)
    write_json(args.output / "gate-control.json", {"epoch": "", "enabled": False})
    started = time.monotonic()
    gen_environment = {
        "SWEEP_METRICS_DIR": str(args.output),
        "SWEEP_INPUT_TOKENS": str(args.input_tokens),
        "PYTHONPATH": str(Path(__file__).parent / "diagnostics") + ":" + os.environ["PYTHONPATH"],
    }
    common_environment = {
        "SWEEP_HTTP_FRESH_CONNECTIONS": "1",
        "SWEEP_HTTP_POLICY_DIR": str(args.output),
        "TRTLLM_DISAGG_NO_RETRY": "1",
        "PYTHONPATH": gen_environment["PYTHONPATH"],
    }
    if args.capacity_gate:
        gen_environment["LONG_PARITY_GATE_PATH"] = str(args.output / "gate-control.json")
    original_popen = disagg_test.popen

    def pinned_popen(command, **kwargs):
        if "--server_role" in command:
            role = command[command.index("--server_role") + 1]
            logical = [2, 3] if role == "generation" else [
                int(index) for index in kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")
            ]
            kwargs["env"] = {**kwargs["env"], "CUDA_VISIBLE_DEVICES": ",".join(
                allocation_devices[index] for index in logical
            )}
        return original_popen(command, **kwargs)

    disagg_test.popen = pinned_popen
    with launch_disaggregated_llm(
        disagg,
        ctx,
        gen,
        str(model),
        server_waiting_timeout=2400,
        request_timeout_s=args.request_timeout,
        request_max_retries=0,
        max_workers=1,
        enable_perf=True,
        extra_env=common_environment,
        gen_extra_env=gen_environment,
    ) as llm:
        response = requests.get(f"{llm.serve_url}/cluster_info", timeout=30)
        response.raise_for_status()
        cluster = response.json()
        workers = cluster["current_workers"]
        write_json(
            args.output / "ready.json",
            {
                "serve_url": llm.serve_url,
                "startup_seconds": time.monotonic() - started,
                "cluster_info": cluster,
                "local_clock": datetime.now().astimezone().isoformat(),
            },
        )
        print(f"SWEEP_READY mode={args.mode} input_tokens={args.input_tokens}", flush=True)
        handled = set()
        while True:
            pending = sorted(p for p in commands.glob("*.json") if p.name not in handled)
            if not pending:
                time.sleep(0.2)
                continue
            command_path = pending[0]
            command = json.loads(command_path.read_text())
            handled.add(command_path.name)
            if command.get("stop"):
                break
            concurrency = command["concurrency"]
            count = command.get("requests", concurrency)
            output_tokens = command.get("output_tokens", args.output_tokens)
            assert count >= concurrency
            assert not command.get("gate") or count == concurrency
            assert not command.get("gate") or args.capacity_gate
            worker_metrics(workers)
            offsets = perf_offsets(perf_directory)
            write_json(
                args.output / "gate-control.json",
                {
                    "epoch": command_path.stem,
                    "enabled": command.get("gate", False),
                    "concurrency": command.get("gate_target", concurrency),
                    "deadline": time.time() + command.get("gate_timeout", 900),
                    "plateau_seconds": command.get("plateau_seconds", 0),
                },
            )
            time.sleep(0.5)
            print(
                f"SWEEP_BATCH {command_path.stem} concurrency={concurrency} requests={count}",
                flush=True,
            )
            profile = None
            if command.get("profile"):
                profile_directory = args.output / "profiles" / command_path.stem
                profile_directory.mkdir(parents=True, exist_ok=True)
                profile = {"directory": str(profile_directory)}
                profile["start"] = control_profile(workers, "start", profile_directory)
            start = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            request_timeout = command.get("request_timeout", args.request_timeout)
            if command.get("gate") and args.input_tokens >= 1_000_000:
                request_timeout = max(request_timeout, concurrency * 2 + 120)
            requests_run = asyncio.run(
                complete_batch(
                    llm.serve_url,
                    str(model),
                    prompt,
                    output_tokens,
                    concurrency,
                    count,
                    request_timeout,
                )
            )
            seconds = time.monotonic() - start
            client_completed_at = datetime.now(timezone.utc).isoformat()
            if profile is not None:
                profile["stop"] = control_profile(workers, "stop", profile_directory)
                previous_sizes = None
                stable = 0
                for _ in range(60):
                    files = list(profile_directory.rglob("*.json"))
                    sizes = {str(path): path.stat().st_size for path in files}
                    stable = stable + 1 if sizes and sizes == previous_sizes else 0
                    if stable >= 6:
                        break
                    previous_sizes = sizes
                    time.sleep(0.5)
                profile["trace_files"] = sizes
            time.sleep(1)
            stats = finish_metrics(
                workers, worker_metrics(workers), sum(r["passed"] for r in requests_run),
                scope=(started_at, client_completed_at, datetime.now().astimezone().isoformat()),
                wake=(llm.serve_url, str(model), prompt),
            )
            gates = [
                json.loads(p.read_text())
                for p in sorted(args.output.glob(f"{command_path.stem}-gate-rank*.json"))
            ]
            write_json(args.output / f"{command_path.stem}-metrics.json", stats)
            write_json(
                args.output / f"{command_path.stem}-perf.json",
                read_new_perf(perf_directory, offsets),
            )
            transfer_directory = args.output / "native-transfer"
            transfer_directory.mkdir(exist_ok=True)
            sources = {
                json.loads(path.read_text())["path"]
                for path in args.output.glob("transfer-source-pid*.json")
            }
            for source in sources:
                for path in Path(source).glob("*.csv"):
                    shutil.copy2(path, transfer_directory / path.name)
            counts = Counter(
                tuple(request["token_ids"]) for request in requests_run if request["passed"]
            )
            passed = [request for request in requests_run if request["passed"]]
            gaps = [
                b - a
                for request in passed
                for a, b in zip(
                    request["token_arrival_seconds"], request["token_arrival_seconds"][1:]
                )
            ]
            result = {
                "mode": args.mode,
                "command": command,
                "started_at_utc": started_at,
                "completed_at_utc": client_completed_at,
                "decode_start_barrier": gates,
                "input_tokens": args.input_tokens,
                "output_tokens": output_tokens,
                "effective_request_timeout_seconds": request_timeout,
                "all_requests_passed": len(passed) == count,
                "completed_requests": len(passed),
                "seconds": seconds,
                "requests_per_second": len(passed) / seconds,
                "output_tokens_per_second": sum(len(r["token_ids"]) for r in passed) / seconds,
                "logical_input_tokens_per_second": len(passed) * args.input_tokens / seconds,
                "unique_outputs": len(counts),
                "ttft_seconds": distribution([r["ttft_seconds"] for r in passed]),
                "latency_seconds": distribution([r["latency_seconds"] for r in passed]),
                "tpot_seconds": distribution([r["tpot_seconds"] for r in passed]),
                "observed_inter_token_gap_seconds": distribution(gaps),
                "output_histogram": [
                    {"token_ids": list(tokens), "count": n} for tokens, n in counts.items()
                ],
                "metrics": summarize_metrics(stats),
                "metrics_rank0_completed_requests": completed_requests(stats),
                "requests": requests_run,
            }
            if profile is not None:
                result["gpu_profile_diagnostic"] = profile
            write_json(args.output / f"{command_path.stem}-results.json", result)
            print(
                f"SWEEP_DONE {command_path.stem} passed={result['all_requests_passed']} "
                f"peak_decode={result['metrics']['peak_concurrent_decode']} "
                f"unique_outputs={len(counts)} seconds={seconds:.2f}",
                flush=True,
            )
    write_json(args.output / "stopped.json", {"seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
