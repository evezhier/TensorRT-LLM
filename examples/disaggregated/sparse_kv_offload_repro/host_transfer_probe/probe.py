#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure same-node NIXL writes and CUDA staging for GPU and host memory."""

import argparse
import json
import mmap
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import zmq

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._torch.disaggregation.base.agent import (
    MemoryDescs,
    MemoryType,
    RegMemoryDescs,
    TransferOp,
    TransferRequest,
)
from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent

HOST_REGISTER_PORTABLE = 0x01
HOST_REGISTER_MAPPED = 0x02


def _cuda_check(result: object, operation: str) -> None:
    status = result[0] if isinstance(result, tuple) else result
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {status}")


def _distribution(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def _layout(
    payload_bytes: int, segments: int, gap_bytes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if payload_bytes <= 0 or segments <= 0 or segments > payload_bytes or gap_bytes < 0:
        raise ValueError("Invalid payload, segment count, or gap")
    sizes = np.full(segments, payload_bytes // segments, dtype=np.int64)
    sizes[: payload_bytes % segments] += 1
    source_offsets = np.zeros(segments, dtype=np.int64)
    source_offsets[1:] = np.cumsum(sizes[:-1])
    destination_offsets = source_offsets + np.arange(segments, dtype=np.int64) * gap_bytes
    span = int(destination_offsets[-1] + sizes[-1])
    return source_offsets, destination_offsets, sizes, span


def _pattern(size: int) -> np.ndarray:
    expected = np.empty(size, dtype=np.uint8)
    for start in range(0, size, 1 << 20):
        stop = min(start + (1 << 20), size)
        expected[start:stop] = (np.arange(start, stop, dtype=np.uint32) * 17 + 31) % 251
    return expected


def _memory_desc(kind: str, pointers: np.ndarray, sizes: np.ndarray) -> MemoryDescs:
    return MemoryDescs.from_arrays_uniform_device(getattr(MemoryType, kind), pointers, sizes, 0)


def _make_agent(name: str) -> NixlTransferAgent:
    return NixlTransferAgent(
        name,
        True,
        num_threads=int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "1")),
        rank=0,
        world_size=1,
    )


def _verify(
    actual: np.ndarray,
    expected: np.ndarray,
    source_offsets: np.ndarray,
    destination_offsets: np.ndarray,
    sizes: np.ndarray,
) -> None:
    for source, destination, size in zip(source_offsets, destination_offsets, sizes, strict=True):
        source, destination, size = int(source), int(destination), int(size)
        if not np.array_equal(
            actual[destination : destination + size], expected[source : source + size]
        ):
            raise AssertionError(
                f"Received bytes differ at source offset {source}, destination offset {destination}"
            )


def _run_sender(args: argparse.Namespace, socket: zmq.Socket) -> None:
    name = f"probe_sender_{os.getpid()}"
    agent = _make_agent(name)
    expected = _pattern(args.payload_bytes)
    source = torch.from_numpy(expected).to("cuda")
    torch.cuda.synchronize()
    host_mapping = None
    host_array = None
    host_ptr = None
    source_kind = "VRAM"
    source_ptr = source.data_ptr()
    if args.mode == "sender_staged":
        host_mapping = mmap.mmap(
            -1, args.payload_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
        )
        host_array = np.frombuffer(host_mapping, dtype=np.uint8)
        host_ptr = int(host_array.ctypes.data)
        _cuda_check(
            cudart.cudaHostRegister(
                host_ptr, args.payload_bytes, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED
            ),
            "cudaHostRegister",
        )
        source_kind = "DRAM"
        source_ptr = host_ptr
    registration = RegMemoryDescs(source_kind, [(source_ptr, args.payload_bytes, 0, name)])
    agent.register_memory(registration)
    published = False
    try:
        published = True
        socket.send_pyobj({"name": name, "agent": bytes(agent.get_local_agent_desc())})
        remote = socket.recv_pyobj()
        agent.load_remote_agent(remote["name"], remote["agent"])
        source_offsets, destination_offsets, sizes, _ = _layout(
            args.payload_bytes, args.segments, args.gap_bytes
        )
        source_ptrs = source_offsets + source_ptr
        destination_ptrs = destination_offsets + remote["ptr"]
        for index in range(args.warmup + args.samples):
            socket.send_pyobj({"ready": index})
            if socket.recv_pyobj() != {"ready": index}:
                raise RuntimeError("Receiver did not prepare its destination")
            copy_seconds = 0.0
            if args.mode == "sender_staged":
                stream = torch.cuda.current_stream().cuda_stream
                copy_start = time.perf_counter()
                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        host_ptr,
                        source.data_ptr(),
                        args.payload_bytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "cudaMemcpyAsync",
                )
                _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
                copy_seconds = time.perf_counter() - copy_start
            request = TransferRequest(
                TransferOp.WRITE,
                _memory_desc(source_kind, source_ptrs, sizes),
                _memory_desc(remote["kind"], destination_ptrs, sizes),
                remote["name"],
                None,
            )
            start = time.perf_counter()
            status = agent.submit_transfer_requests(request)
            completed = status.wait(timeout_ms=args.timeout_ms)
            elapsed = time.perf_counter() - start
            if not completed or not status.is_completed():
                raise RuntimeError(f"NIXL completion unproven: {status.last_status_str()}")
            socket.send_pyobj(
                {
                    "sample": index,
                    "transfer_seconds": elapsed,
                    "local_copy_seconds": copy_seconds,
                }
            )
            reply = socket.recv_pyobj()
            if reply != {"verified": index}:
                raise RuntimeError(f"Receiver did not verify sample {index}: {reply!r}")
        socket.send_pyobj("done")
        if socket.recv_pyobj() != "done":
            raise RuntimeError("Receiver did not acknowledge completion")
    except BaseException:
        traceback.print_exc()
        if published:
            sys.stderr.flush()
            os._exit(3)
        raise
    agent.deregister_memory(registration)
    agent.shutdown()
    if host_ptr is not None:
        _cuda_check(cudart.cudaHostUnregister(host_ptr), "cudaHostUnregister")
        del host_array
        host_mapping.close()


def _run_receiver(args: argparse.Namespace, socket: zmq.Socket) -> None:
    name = f"probe_receiver_{os.getpid()}"
    source_offsets, destination_offsets, sizes, span = _layout(
        args.payload_bytes, args.segments, args.gap_bytes
    )
    expected = _pattern(args.payload_bytes)
    host_mapping = None
    host_array = None
    host_ptr = None
    gpu = None
    if args.mode in ("host", "staged", "sender_staged"):
        host_mapping = mmap.mmap(-1, span, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        host_array = np.frombuffer(host_mapping, dtype=np.uint8)
        host_ptr = int(host_array.ctypes.data)
        _cuda_check(
            cudart.cudaHostRegister(host_ptr, span, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED),
            "cudaHostRegister",
        )
    if args.mode in ("gpu", "staged"):
        gpu = torch.empty(span, dtype=torch.uint8, device="cuda")
    kind = "DRAM" if args.mode in ("host", "sender_staged") else "VRAM"
    target = host_ptr if kind == "DRAM" else gpu.data_ptr()
    agent = _make_agent(name)
    registration = RegMemoryDescs(kind, [(target, span, 0, name)])
    agent.register_memory(registration)
    published = False
    try:
        remote = socket.recv_pyobj()
        published = True
        socket.send_pyobj(
            {
                "name": name,
                "agent": bytes(agent.get_local_agent_desc()),
                "ptr": target,
                "kind": kind,
            }
        )
        agent.load_remote_agent(remote["name"], remote["agent"])
        rows = []
        for index in range(args.warmup + args.samples):
            if socket.recv_pyobj() != {"ready": index}:
                raise RuntimeError("Sender sample sequence mismatch")
            if host_array is not None:
                host_array.fill(0)
            if gpu is not None:
                gpu.zero_()
                torch.cuda.synchronize()
            socket.send_pyobj({"ready": index})
            result = socket.recv_pyobj()
            if result.get("sample") != index:
                raise RuntimeError("Sender completion sequence mismatch")
            copy_seconds = result["local_copy_seconds"]
            if args.mode == "staged":
                stream = torch.cuda.current_stream().cuda_stream
                start = time.perf_counter()
                for destination, size in zip(destination_offsets, sizes, strict=True):
                    offset, count = int(destination), int(size)
                    _cuda_check(
                        cudart.cudaMemcpyAsync(
                            host_ptr + offset,
                            gpu.data_ptr() + offset,
                            count,
                            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                            stream,
                        ),
                        "cudaMemcpyAsync",
                    )
                _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
                copy_seconds = time.perf_counter() - start
            actual = gpu.cpu().numpy() if args.mode == "gpu" else host_array
            _verify(actual, expected, source_offsets, destination_offsets, sizes)
            transfer_seconds = result["transfer_seconds"]
            rows.append(
                {
                    "sample": index,
                    "warmup": index < args.warmup,
                    "transfer_seconds": transfer_seconds,
                    "local_copy_seconds": copy_seconds,
                    "total_seconds": transfer_seconds + copy_seconds,
                    "verified_bytes": args.payload_bytes,
                }
            )
            socket.send_pyobj({"verified": index})
        if socket.recv_pyobj() != "done":
            raise RuntimeError("Sender did not complete the case")
        socket.send_pyobj("done")
        measured = rows[args.warmup :]
        report = {
            "mode": args.mode,
            "payload_bytes": args.payload_bytes,
            "segments": args.segments,
            "gap_bytes": args.gap_bytes,
            "registered_bytes": span,
            "num_threads": int(os.environ.get("TRTLLM_NIXL_NUM_THREADS", "1")),
            "ucx_tls": os.environ.get("UCX_TLS"),
            "all_samples_verified": True,
            "transfer_ms": _distribution([r["transfer_seconds"] * 1000 for r in measured]),
            "local_copy_ms": _distribution([r["local_copy_seconds"] * 1000 for r in measured]),
            "total_ms": _distribution([r["total_seconds"] * 1000 for r in measured]),
            "samples": rows,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key != "samples"}),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        if published:
            sys.stderr.flush()
            os._exit(3)
        raise
    agent.deregister_memory(registration)
    agent.shutdown()
    if host_ptr is not None:
        _cuda_check(cudart.cudaHostUnregister(host_ptr), "cudaHostUnregister")
    if host_array is not None:
        del actual
        del host_array
        host_mapping.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("sender", "receiver"), required=True)
    parser.add_argument("--mode", choices=("gpu", "host", "staged", "sender_staged"), required=True)
    parser.add_argument("--payload-bytes", type=int, default=109924352)
    parser.add_argument("--segments", type=int, default=1284)
    parser.add_argument("--gap-bytes", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.warmup < 0 or args.timeout_ms <= 0:
        parser.error("samples and timeout must be positive; warmup must be nonnegative")
    if args.role == "receiver" and args.output is None:
        parser.error("receiver requires --output")
    _layout(args.payload_bytes, args.segments, args.gap_bytes)
    torch.cuda.set_device(0)
    context = zmq.Context()
    socket = context.socket(zmq.REQ if args.role == "sender" else zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms + 10000)
    socket.setsockopt(zmq.SNDTIMEO, args.timeout_ms + 10000)
    if args.role == "sender":
        socket.connect(f"tcp://127.0.0.1:{args.port}")
    else:
        socket.bind(f"tcp://127.0.0.1:{args.port}")
    try:
        if args.role == "sender":
            _run_sender(args, socket)
        else:
            _run_receiver(args, socket)
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
