#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure CUDA writes to host memory shared with a separate process."""

import argparse
import contextlib
import ctypes
import hashlib
import json
import mmap
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

DEFAULT_PAYLOAD_BYTES = 109924352
DEFAULT_SEGMENTS = 1284
DEFAULT_GAP_BYTES = 4096
HOST_REGISTER_PORTABLE = 1
HOST_REGISTER_MAPPED = 2
SCATTER_BLOCK_BYTES = 16384


if triton is not None:

    @triton.jit
    def _scatter_kernel(
        src, dst_addr, BASE: tl.constexpr, EXTRA: tl.constexpr,
        GAP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        segment = tl.program_id(0)
        tile = tl.program_id(1)
        length = tl.where(segment < EXTRA, BASE + 1, BASE)
        source_offset = segment * BASE + tl.minimum(segment, EXTRA)
        positions = tile * BLOCK + tl.arange(0, BLOCK)
        destination = dst_addr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
        values = tl.load(src + source_offset + positions, mask=positions < length, other=0)
        tl.store(
            destination + source_offset + segment * GAP + positions,
            values,
            mask=positions < length,
        )

    @triton.jit
    def _host_pool_scatter_kernel(
        staging_addr, pool_ptrs, BASE: tl.constexpr, EXTRA: tl.constexpr,
        GAP: tl.constexpr, BLOCK: tl.constexpr,
    ):
        segment = tl.program_id(0)
        tile = tl.program_id(1)
        length = tl.where(segment < EXTRA, BASE + 1, BASE)
        source_offset = segment * BASE + tl.minimum(segment, EXTRA) + segment * GAP
        positions = tile * BLOCK + tl.arange(0, BLOCK)
        source = staging_addr.to(tl.uint64).to(tl.pointer_type(tl.uint8))
        destination = tl.load(pool_ptrs + segment).to(tl.uint64).to(tl.pointer_type(tl.uint8))
        values = tl.load(source + source_offset + positions, mask=positions < length, other=0)
        tl.store(destination + positions, values, mask=positions < length)


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


def _worker(fd: int, payload_bytes: int, segments: int, gap_bytes: int, method: str) -> None:
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
        if method == "scatter_kernel":
            if triton is None:
                raise RuntimeError("scatter_kernel requires Triton in the container")
            result = cudart.cudaHostGetDevicePointer(host_ptr, 0)
            _cuda_check(result, "cudaHostGetDevicePointer")
            device_host_ptr = int(result[1])
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
            if method == "scatter_kernel":
                base = payload_bytes // segments
                extra = payload_bytes % segments
                _scatter_kernel[
                    (segments, triton.cdiv(base + 1, SCATTER_BLOCK_BYTES))
                ](
                    source, device_host_ptr, base, extra, gap_bytes,
                    SCATTER_BLOCK_BYTES,
                )
            else:
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
        if gap_bytes and index + 1 < len(sizes):
            gap = actual[dst + int(size):dst + int(size) + gap_bytes]
            if not np.all(gap == 0xA5):
                raise AssertionError(f"copy modified the gap after segment {index}")


@contextlib.contextmanager
def _kvcm_host_pool(payload_bytes: int, segments: int):
    import torch

    from tensorrt_llm.runtime.kv_cache_manager_v2 import (
        BAD_PAGE_INDEX,
        AttentionLayerConfig,
        BatchDesc,
        BufferConfig,
        CacheTier,
        GpuCacheTierConfig,
        KVCacheDesc,
        KVCacheManager,
        KVCacheManagerConfig,
        SparseCacheConfig,
    )

    stream = torch.cuda.current_stream().cuda_stream
    tokens_per_block = 128
    tokens = segments * tokens_per_block
    page_bytes = ((payload_bytes + segments - 1) // segments + 255) // 256 * 256
    manager = KVCacheManager(
        KVCacheManagerConfig(
            tokens_per_block=tokens_per_block,
            cache_tiers=[GpuCacheTierConfig(64 << 20)],
            constraints=[BatchDesc(kv_caches=[KVCacheDesc(capacity=tokens, history_length=0)])],
            layers=[
                AttentionLayerConfig(
                    0,
                    [BufferConfig("key", page_bytes, is_sparse=True), BufferConfig("index", 256)],
                    sparse_cache=SparseCacheConfig(
                        num_selected_blocks=2, max_cached_units=4,
                        unit_tokens=tokens_per_block,
                    ),
                )
            ],
            host_sparse_pool_quota=256 << 20,
        )
    )
    cache = None
    try:
        if manager.sparse_backing_medium() != CacheTier.HOST_MEM:
            raise RuntimeError("KVCM2 sparse key pool is not host-backed")
        cache = manager.create_kv_cache()
        if not cache.resume(stream) or not cache.resize(tokens, tokens):
            raise RuntimeError("KVCM2 could not allocate the sparse receive pages")
        group = manager.get_layer_group_id(0)
        slots = list(cache.get_base_page_indices(group))
        if len(slots) < segments or any(slot == BAD_PAGE_INDEX for slot in slots[:segments]):
            raise RuntimeError("KVCM2 sparse receive pages are incomplete")
        base = int(manager.get_mem_pool_base_address(0, "key"))
        stride = int(manager.get_page_stride(0, "key"))
        if stride < page_bytes:
            raise RuntimeError("KVCM2 sparse pool stride is smaller than a page")
        yield [base + int(slot) * stride for slot in slots[:segments]], stride
    finally:
        if cache is not None:
            cache.close()
        manager.shutdown()


def _case(
    args: argparse.Namespace, segments: int, gap_bytes: int, method: str,
    pool: tuple[list[int], int] | None = None,
    pool_copy_method: str = "cpu_memcpy",
) -> dict:
    offsets, sizes, span = _layout(args.payload_bytes, segments, gap_bytes)
    expected = np.arange(args.payload_bytes, dtype=np.uint8)
    fd = os.memfd_create("trtllm-shared-host-copy", 0)
    os.ftruncate(fd, span)
    mapping = mmap.mmap(fd, span, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE)
    actual = np.frombuffer(mapping, dtype=np.uint8)
    staging_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", "--fd", str(fd),
        "--payload-bytes", str(args.payload_bytes), "--segments", str(segments),
        "--gap-bytes", str(gap_bytes), "--method", method,
    ]
    worker = subprocess.Popen(
        command, pass_fds=(fd,), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    receiver_registered = False
    try:
        if pool is not None and pool_copy_method == "gpu_scatter":
            import torch

            try:
                from cuda.bindings import runtime as cudart
            except ImportError:
                from cuda import cudart

            _cuda_check(
                cudart.cudaHostRegister(
                    staging_ptr, span, HOST_REGISTER_PORTABLE | HOST_REGISTER_MAPPED
                ),
                "cudaHostRegister(receiver staging)",
            )
            receiver_registered = True
            result = cudart.cudaHostGetDevicePointer(staging_ptr, 0)
            _cuda_check(result, "cudaHostGetDevicePointer(receiver staging)")
            device_staging_ptr = int(result[1])
            pool_device_ptrs = []
            for pointer in pool[0]:
                result = cudart.cudaHostGetDevicePointer(pointer, 0)
                _cuda_check(result, "cudaHostGetDevicePointer(KVCM2 pool page)")
                pool_device_ptrs.append(int(result[1]))
            pool_ptrs_cuda = torch.tensor(pool_device_ptrs, dtype=torch.int64, device="cuda")
            torch.cuda.synchronize()
        assert worker.stdin is not None and worker.stdout is not None
        ready_line = worker.stdout.readline()
        if not ready_line:
            raise RuntimeError(f"CUDA worker exited during initialization: {worker.wait()}")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError(f"CUDA worker did not initialize: {ready}")
        samples = []
        handoff_samples = []
        pool_copy_samples = []
        for index in range(args.warmup + args.samples):
            actual.fill(0xA5)
            if pool is not None:
                pointers, _ = pool
                for pointer, size in zip(pointers, sizes, strict=True):
                    ctypes.memset(pointer, 0xA5, int(size))
            handoff_start = time.perf_counter()
            worker.stdin.write("copy\n")
            worker.stdin.flush()
            result_line = worker.stdout.readline()
            if not result_line:
                raise RuntimeError(f"CUDA worker exited during copy: {worker.wait()}")
            result = json.loads(result_line)
            copy_start = time.perf_counter()
            if pool is not None:
                pointers, _ = pool
                if pool_copy_method == "gpu_scatter":
                    base = args.payload_bytes // segments
                    extra = args.payload_bytes % segments
                    _host_pool_scatter_kernel[
                        (segments, triton.cdiv(base + 1, SCATTER_BLOCK_BYTES))
                    ](
                        device_staging_ptr, pool_ptrs_cuda, base, extra, gap_bytes,
                        SCATTER_BLOCK_BYTES,
                    )
                    torch.cuda.synchronize()
                else:
                    for segment, (pointer, offset, size) in enumerate(
                        zip(pointers, offsets, sizes, strict=True)
                    ):
                        ctypes.memmove(
                            pointer, staging_ptr + int(offset) + segment * gap_bytes, int(size)
                        )
            pool_copy_ms = (time.perf_counter() - copy_start) * 1000
            handoff_ms = (time.perf_counter() - handoff_start) * 1000
            _verify(actual, expected, offsets, sizes, gap_bytes)
            if pool is not None:
                for segment, (pointer, offset, size) in enumerate(
                    zip(pointers, offsets, sizes, strict=True)
                ):
                    received = np.ctypeslib.as_array(
                        (ctypes.c_uint8 * int(size)).from_address(pointer)
                    )
                    if not np.array_equal(received, expected[int(offset):int(offset) + int(size)]):
                        raise AssertionError(f"KVCM2 host page differs at segment {segment}")
            if index >= args.warmup:
                samples.append(result["copy_ms"])
                if pool is not None:
                    handoff_samples.append(handoff_ms)
                    pool_copy_samples.append(pool_copy_ms)
        worker.stdin.write("stop\n")
        worker.stdin.flush()
        if worker.wait(timeout=30):
            raise RuntimeError(f"CUDA worker exited with status {worker.returncode}")
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
        if receiver_registered:
            _cuda_check(cudart.cudaHostUnregister(staging_ptr), "cudaHostUnregister(receiver staging)")
        if worker.stdin is not None:
            worker.stdin.close()
        if worker.stdout is not None:
            worker.stdout.close()
        del actual
        mapping.close()
        os.close(fd)
    ordered = sorted(samples)
    report = {
        "gpu": ready["gpu"], "method": method, "payload_bytes": args.payload_bytes,
        "segments": segments, "gap_bytes": gap_bytes, "span_bytes": span,
        "samples_ms": samples, "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(ordered), "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "verified": True,
    }
    if pool is not None:
        report.update(
            host_pool="KVCM2 sparse host-backed pool",
            host_pool_copy_method=pool_copy_method,
            host_pool_stride_bytes=pool[1],
            handoff_samples_ms=handoff_samples,
            handoff_p50_ms=statistics.median(handoff_samples),
            host_pool_copy_samples_ms=pool_copy_samples,
            host_pool_copy_p50_ms=statistics.median(pool_copy_samples),
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fd", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--payload-bytes", type=int, default=DEFAULT_PAYLOAD_BYTES)
    parser.add_argument("--segments", type=int, default=DEFAULT_SEGMENTS)
    parser.add_argument("--gap-bytes", type=int, default=DEFAULT_GAP_BYTES)
    parser.add_argument("--method", choices=("cuda_memcpy", "scatter_kernel"),
                        default="cuda_memcpy", help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--kvcm", action="store_true", help="Copy into a real KVCM2 sparse host pool")
    args = parser.parse_args()
    if args.worker:
        _worker(args.fd, args.payload_bytes, args.segments, args.gap_bytes, args.method)
        return
    if args.samples <= 0 or args.warmup < 0:
        parser.error("samples must be positive and warmup must be nonnegative")
    report = {
        "mechanism": "cross-process shared host mapping with CUDA copy and scatter",
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }

    def publish() -> None:
        output = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output)
        print(output, end="", flush=True)

    for segments, gap_bytes, method in (
        (1, 0, "cuda_memcpy"),
        (args.segments, 0, "cuda_memcpy"),
        (args.segments, args.gap_bytes, "cuda_memcpy"),
        (args.segments, 0, "scatter_kernel"),
        (args.segments, args.gap_bytes, "scatter_kernel"),
    ):
        report["cases"].append(_case(args, segments, gap_bytes, method))
        publish()
    if args.kvcm:
        try:
            with _kvcm_host_pool(args.payload_bytes, args.segments) as pool:
                report["cases"].append(
                    _case(args, args.segments, args.gap_bytes, "scatter_kernel", pool)
                )
                publish()
                report["cases"].append(
                    _case(
                        args, args.segments, args.gap_bytes, "scatter_kernel", pool,
                        "gpu_scatter",
                    )
                )
        except Exception as error:
            report["kvcm_error"] = f"{type(error).__name__}: {error}"
            publish()
            raise
        publish()


if __name__ == "__main__":
    main()
