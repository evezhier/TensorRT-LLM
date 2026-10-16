# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic two-node PD handoff through the production Python transceiver."""

import argparse
import ctypes
import json
import os
import time

import torch
import yaml
import zmq
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import create_kv_cache_transceiver
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping

from run_cache_transceiver_test import (
    AttentionTypeCpp,
    build_kv_cache_manager,
    free_sequence,
    make_request,
)

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def _check_cuda(result, operation):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {result[0]}")


def _pattern(group, pool, ordinal):
    return (group * 31 + pool * 17 + ordinal * 47 + 1) % 251


def _pages(manager, transceiver, rid, prompt_blocks):
    cache = manager.kv_cache_map[rid]
    page_table = transceiver._page_table
    for group, layer_group in enumerate(page_table.layer_groups):
        slots = cache.get_base_page_indices(group)
        if len(slots) < prompt_blocks:
            raise RuntimeError(f"group {group} has only {len(slots)} prompt pages")
        pools = page_table.pool_groups[layer_group.pool_group_idx].pools
        for ordinal in range(prompt_blocks):
            slot = int(slots[ordinal])
            if slot < 0:
                raise RuntimeError(f"group {group} is missing prompt block {ordinal}")
            for pool_idx in range(len(pools)):
                pool = get_physical_pool(page_table, group, pool_idx)
                address = pool.base_address + slot * pool.slot_stride_bytes
                yield group, pool_idx, ordinal, pool, address


def _fill(manager, transceiver, rid, prompt_blocks):
    stream = torch.cuda.current_stream().cuda_stream
    for group, pool_idx, ordinal, pool, address in _pages(
        manager, transceiver, rid, prompt_blocks
    ):
        value = _pattern(group, pool_idx, ordinal)
        if pool.memory_type == "DRAM":
            ctypes.memset(address, value, pool.slot_bytes)
        else:
            _check_cuda(
                cudart.cudaMemsetAsync(address, value, pool.slot_bytes, stream),
                "cudaMemsetAsync",
            )
    torch.cuda.current_stream().synchronize()


def _verify(manager, transceiver, rid, prompt_blocks):
    payload_bytes = 0
    for group, pool_idx, ordinal, pool, address in _pages(
        manager, transceiver, rid, prompt_blocks
    ):
        expected = _pattern(group, pool_idx, ordinal)
        if pool.memory_type == "DRAM":
            payload = ctypes.string_at(address, pool.slot_bytes)
        else:
            host = (ctypes.c_ubyte * pool.slot_bytes)()
            _check_cuda(
                cudart.cudaMemcpy(
                    ctypes.addressof(host),
                    address,
                    pool.slot_bytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                ),
                "cudaMemcpyDeviceToHost",
            )
            payload = bytes(host)
        if payload != bytes([expected]) * pool.slot_bytes:
            raise RuntimeError(
                f"byte mismatch at group={group} pool={pool_idx} ordinal={ordinal}"
            )
        payload_bytes += pool.slot_bytes
    return payload_bytes


def _pool_footprint(transceiver):
    footprint = {"VRAM": 0, "DRAM": 0}
    for group in transceiver._page_table.pool_groups:
        for pool in group.pools:
            footprint[pool.memory_type] += pool.num_slots * pool.slot_stride_bytes
    return footprint


def _expect_done(request, status, role):
    if role == "ctx":
        completed, failed = status
    else:
        completed, failed, cancelled = status
        if cancelled:
            raise RuntimeError(f"generation transfer cancelled: {cancelled}")
    ids = {item.py_request_id if hasattr(item, "py_request_id") else item for item in completed}
    if request.py_request_id not in ids or failed:
        raise RuntimeError(f"{role} transfer did not complete: {status}")


def _run_request(role, socket, manager, transceiver, mode, sample, tokens, tpb):
    rid = sample + 1
    request = make_request(role == "ctx", rid, tokens, "PYTHON") if role == "ctx" else None
    if role == "ctx":
        if not manager.prepare_context(request) or not manager.resize_context(request, tokens):
            raise RuntimeError("context KV allocation failed")
        _fill(manager, transceiver, rid, tokens // tpb)
        # Model execution normally advances this watermark after writing KV.
        if not manager.kv_cache_map[rid].resize(None, tokens):
            raise RuntimeError("could not mark synthetic context history complete")
        request.py_disaggregated_params.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
        if socket.recv() != b"go":
            raise RuntimeError("generation peer did not request context metadata")
        params = trtllm.ContextPhaseParams(
            first_gen_tokens=[],
            req_id=rid,
            opaque_state=None,
            draft_tokens=None,
            ctx_dp_rank=0,
            disagg_info_endpoint=transceiver._context_info_endpoint,
        )
        socket.send_pyobj(params)
        if socket.recv() != b"ready":
            raise RuntimeError("generation peer did not publish its destinations")
        deadline = time.monotonic() + 30
        while not transceiver._transfer_worker._sender._is_req_ready(rid, 1):
            if time.monotonic() >= deadline:
                raise TimeoutError("context sender did not receive generation page metadata")
            time.sleep(0.001)
        start = time.perf_counter()
        transceiver.respond_and_send_async(request)
        socket.send(b"sent")
        _expect_done(request, transceiver.check_context_transfer_status(None), role)
        elapsed = time.perf_counter() - start
        if socket.recv() != b"done":
            raise RuntimeError("generation peer did not verify the transfer")
        socket.send(b"ack")
        free_sequence(manager, request, None, True)
        return elapsed, None

    socket.send(b"go")
    params = socket.recv_pyobj()
    request = make_request(False, rid, tokens, "PYTHON", ctx_params=params)
    request.py_disaggregated_params.schedule_style = DisaggScheduleStyle.GENERATION_FIRST
    if not manager.prepare_disagg_gen_init(request):
        raise RuntimeError("generation KV allocation failed")
    start = time.perf_counter()
    transceiver.request_and_receive_async(request)
    socket.send(b"ready")
    if socket.recv() != b"sent":
        raise RuntimeError("context peer did not submit the transfer")
    _expect_done(request, transceiver.check_gen_transfer_status(None), role)
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    payload_bytes = _verify(manager, transceiver, rid, tokens // tpb)
    socket.send(b"done")
    if socket.recv() != b"ack":
        raise RuntimeError("context peer did not finish ownership handoff")
    free_sequence(manager, request, None, True)
    return elapsed, payload_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=("ctx", "gen"))
    args = parser.parse_args()
    role = args.role
    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError("the first PD host-aware milestone uses one GPU per node")
    torch.cuda.set_device(0)
    with open(os.environ["CTT_CONFIG"]) as file:
        cfg = yaml.safe_load(file)
    benchmark = cfg["pd_host_aware_benchmark"]
    tpb = int(cfg["kv_cache"]["tokens_per_block"])
    tokens = int(benchmark["prompt_tokens"])
    if tokens <= 0 or tokens % tpb:
        raise ValueError("prompt_tokens must be positive and block-aligned")
    modes = benchmark.get("modes", ["gpu", "host"])
    if any(mode not in ("gpu", "host") for mode in modes):
        raise ValueError("PD benchmark modes must be gpu or host")
    config = dict(cfg["kv_cache"], _max_request_len=tokens)
    if int(config["num_layers"]) < 3:
        raise ValueError("PD benchmark needs multiple sparse layers and a dense layer")
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1, gpus_per_node=1)
    distributed = Distributed.get(mapping)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP if role == "ctx" else zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(benchmark.get("timeout_ms", 30000)))
    socket.setsockopt(zmq.SNDTIMEO, int(benchmark.get("timeout_ms", 30000)))
    port = int(os.environ["ZMQ_PORT"])
    if role == "ctx":
        socket.bind(f"tcp://*:{port}")
    else:
        socket.connect(f"tcp://{os.environ['CTX_NODE']}:{port}")
    output = os.path.join(cfg["environment"]["work_dir"], "pd_host_aware")
    os.makedirs(output, exist_ok=True)
    output_file = os.path.join(output, f"{role}_sweep{os.environ.get('CTT_SWEEP', '0')}.jsonl")
    with open(output_file, "w"):
        pass
    os.environ["TRTLLM_DISAGG_NO_RETRY"] = "1"
    os.environ["TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP"] = "0"
    for mode_idx, mode in enumerate(modes):
        print(f"[CTT_CASE_BEGIN] ci={mode_idx} label=pd/{mode}", flush=True)
        os.environ["TRTLLM_ENABLE_HOST_AWARE_KV_TRANSFER"] = "1" if mode == "host" else "0"
        manager = build_kv_cache_manager(
            config, mapping, True, host_sparse=mode == "host", is_disagg=True
        )
        cache_config = CacheTransceiverConfig(
            backend="NIXL",
            transceiver_runtime="PYTHON",
            kv_cache_bounce_size_mb=0,
            kv_transfer_timeout_ms=int(benchmark.get("timeout_ms", 30000)),
            enable_pipelined_transfer=False,
        )
        transceiver = create_kv_cache_transceiver(
            mapping, distributed, manager, AttentionTypeCpp.DEFAULT, cache_config
        )
        footprint = _pool_footprint(transceiver)
        try:
            for sample in range(int(benchmark.get("warmup", 1)) + int(benchmark.get("samples", 3))):
                torch.cuda.reset_peak_memory_stats()
                elapsed, payload_bytes = _run_request(
                    role, socket, manager, transceiver, mode, sample, tokens, tpb
                )
                if sample >= int(benchmark.get("warmup", 1)):
                    record = {
                        "mode": mode,
                        "sweep": int(os.environ.get("CTT_SWEEP", "0")),
                        "role": role,
                        "sample": sample,
                        "payload_bytes": payload_bytes,
                        "handoff_seconds": elapsed,
                        "effective_gbps": payload_bytes / elapsed / 1e9 if payload_bytes else None,
                        "torch_peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                        "registered_gpu_pool_bytes": footprint["VRAM"],
                        "registered_host_pool_bytes": footprint["DRAM"],
                        "verified": role == "gen",
                    }
                    with open(output_file, "a") as file:
                        file.write(json.dumps(record) + "\n")
                    print(json.dumps(record), flush=True)
        finally:
            transceiver.shutdown()
            manager.shutdown()
    socket.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        tensorrt_llm.logger.error("PD handoff benchmark failed; preserving ownership until exit")
        raise
