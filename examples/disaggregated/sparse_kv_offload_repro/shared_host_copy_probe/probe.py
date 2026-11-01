#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure CUDA writes to host memory shared with a separate process."""

import argparse
import ctypes
import json
import mmap
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_PAYLOAD_BYTES = 109924352
DEFAULT_SEGMENTS = 1284
DEFAULT_GAP_BYTES = 4096
HOST_REGISTER_PORTABLE = 1
HOST_REGISTER_MAPPED = 2


def _layout(payload_bytes: int, segments: int, gap_bytes: int) -> tuple[np.ndarray, np.ndarray, int]:
    if payload_bytes <= 0 or segments <= 0 or segments > payload_bytes or gap_bytes < 0:
        raise ValueError("payload_bytes, segments, and gap_bytes must describe a valid layout")
    sizes = np.full(segments, payload_bytes // segments, dtype=np.int64)
    sizes[: payload_bytes % segments] += 1
    offsets = np.zeros(segments, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes[:-1])
    span = payload_bytes + (segments - 1) * gap_bytes
    return offsets, sizes, span


def _cuda_check(result: object, operation: str) -> None:
    status = result[0] if isinstance(result, tuple) else result
    if status != 0:
        raise RuntimeError(f"{operation} failed: {status}")


def _worker(fd: int, payload_bytes: int, segments: int, gap_bytes: int) -> None:
    import torch

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        from cuda import cudart

    offsets, sizes, span = _layout(payload_bytes, segments, gap_bytes)
    mapping = mmap.mmap(fd, span, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    _cuda_check(
        cudart.cudaHostRegister(
            host_ptr, span, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED
        ),
        "cudaHostRegister(shared host mapping)",
    )
    try:
        expected = np.arange(payload_bytes, dtype=np.uint8)
        source = torch.from_numpy(expected).to("cuda")
        torch.cuda.synchronize()
        stream = torch.cuda.current_stream().cuda_stream
        print(json.dumps({"ready": True, "gpu": torch.cuda.get_device_name()}), flush=True)
        for line in sys.stdin:
            if line.strip() == "stop":
                break
            if line.strip() != "copy":
                raise ValueError(f"unexpected command: {line.strip()!r}")
            start = time.perf_counter()
            for index, (offset, size) in enumerate(zip(offsets, sizes, strict=True)):
                offset_int = int(offset)
                _cuda_check(
                    cudart.cudaMemcpyAsync(
                        host_ptr + offset_int + index * gap_bytes,
                        source.data_ptr() + offset_int,
                        int(size),
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        stream,
                    ),
                    "cudaMemcpyAsync(D2H)",
                )
            _cuda_check(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            print(json.dumps({"copy_ms": (time.perf_counter() - start) * 1000}), flush=True)
    finally:
        _cuda_check(cudart.cudaHostUnregister(host_ptr), "cudaHostUnregister")
        mapping.close()


def _verify(actual: np.ndarray, expected: np.ndarray, offsets: np.ndarray,
            sizes: np.ndarray, gap_bytes: int) -> None:
    for index, (offset, size) in enumerate(zip(offsets, sizes, strict=True)):
        src = int(offset)
        dst = src + index * gap_bytes
        if not np.array_equal(actual[dst:dst + int(size)], expected[src:src + int(size)]):
            raise AssertionError(f"copy differs at segment {index}, destination offset {dst}")


def _case(args: argparse.Namespace, segments: int, gap_bytes: int) -> dict:
    offsets, sizes, span = _layout(args.payload_bytes, segments, gap_bytes)
    expected = np.arange(args.payload_bytes, dtype=np.uint8)
    fd = os.memfd_create("trtllm-shared-host-copy", 0)
    os.ftruncate(fd, span)
    mapping = mmap.mmap(fd, span, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    actual = np.frombuffer(mapping, dtype=np.uint8)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--fd", str(fd),
        "--payload-bytes", str(args.payload_bytes), "--segments", str(segments),
        "--gap-bytes", str(gap_bytes),
    ]
    worker = subprocess.Popen(
        command, pass_fds=(fd,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert worker.stdin is not None and worker.stdout is not None
        ready_line = worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"CUDA worker exited during initialization: {worker.wait()}")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"CUDA worker did not initialize: {ready}")
        samples = []
        for index in range(args.warmup + args.samples):
            actual.fill(0xA5)
            worker.stdin.write("copy\n")
            worker.stdin.flush()
            result_line = worker.stdout.readline()
            if not result_line:
                raise RuntimeError(f"CUDA worker exited during copy: {worker.wait()}")
            result = json.loads(result_line)
            _verify(actual, expected, offsets, sizes, gap_bytes)
            if index >= args.warmup:
                samples.append(result["copy_ms"])
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        if worker.wait(timeout=30):
            raise RuntimeError(f"CUDA worker exited with status {worker.returncode}")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if worker.stdin is not None:
            worker.stdin.close()
        if worker.stdout is not None:
            worker.stdout.close()
        del actual
        mapping.close()
        os.close(fd)
    ordered = sorted(samples)
    return {
        "gpu": ready["gpu"], "payload_bytes": args.payload_bytes,
        "segments": segments, "gap_bytes": gap_bytes, "span_bytes": span,
        "samples_ms": samples, "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(ordered), "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fd", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--payload-bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--gap-bytes", type=int, default=DEFAULT_GAP_BYTES)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        _worker(args.fd, args.payload_bytes, args.segments, args.gap_bytes)
        return
    if args.samples <= 0 or args.warmup < 0:
        parser.error("samples must be positive and warmup must be nonnegative")
    cases = [
        _case(args, 1, 0),
        _case(args, args.segments, 0),
        _case(args, args.segments, args.gap_bytes),
    ]
    report = {"mechanism": "cross-process shared host mapping with CUDA D2H", "cases": cases}
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
