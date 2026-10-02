# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XpuCopyBackend: GPU<->CPU block copies on Intel XPU through swap_blocks_batch."""

from __future__ import annotations

import time

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_xpu():
    pytest.skip("Requires Intel XPU", allow_module_level=True)

from vllm.v1.simple_kv_offload.copy_backend import XpuCopyBackend

NUM_BLOCKS = 32


def _caches(device: torch.device) -> tuple[dict, dict]:
    # Two layers with different block sizes, like a hybrid model's groups.
    gpu = {
        "a": torch.randint(-128, 127, (NUM_BLOCKS, 4096), dtype=torch.int8, device=device),
        "b": torch.randint(-128, 127, (NUM_BLOCKS, 1536), dtype=torch.int8, device=device),
    }
    pool = torch.zeros(NUM_BLOCKS * (4096 + 1536), dtype=torch.int8, pin_memory=True)
    cpu = {
        "a": pool[: NUM_BLOCKS * 4096].view(NUM_BLOCKS, 4096),
        "b": pool[NUM_BLOCKS * 4096 :].view(NUM_BLOCKS, 1536),
    }
    return gpu, cpu


def _wait(events: list, event_idx: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not any(i == event_idx for i, _ in events):
        assert time.monotonic() < deadline, "copy never completed"
        time.sleep(0.001)
    for i, event in events:
        if i == event_idx:
            event.synchronize()


def test_store_then_load_roundtrip():
    device = torch.device("xpu", 0)
    gpu, cpu = _caches(device)
    original = {k: v.clone() for k, v in gpu.items()}
    backend = XpuCopyBackend()
    backend.init(gpu, cpu, device, current_platform.Stream(), current_platform.Stream())
    try:
        stores: list = []
        backend.launch_copy([3, 7, 9], [0, 1, 2], is_store=True, event_idx=0, events_list=stores)
        _wait(stores, 0)
        for k in gpu:
            assert torch.equal(cpu[k][[0, 1, 2]], original[k][[3, 7, 9]].cpu())

        for k in gpu:
            gpu[k].zero_()
        torch.xpu.synchronize()
        loads: list = []
        backend.launch_copy([0, 1, 2], [10, 11, 12], is_store=False, event_idx=0, events_list=loads)
        _wait(loads, 0)
        for k in gpu:
            assert torch.equal(gpu[k][[10, 11, 12]].cpu(), original[k][[3, 7, 9]].cpu())
            assert torch.count_nonzero(gpu[k][[0, 1, 2]]) == 0
    finally:
        backend.shutdown()


def test_store_waits_for_compute_write():
    """A store ordered after a compute-stream event copies the written bytes."""
    device = torch.device("xpu", 0)
    gpu, cpu = _caches(device)
    backend = XpuCopyBackend()
    backend.init(gpu, cpu, device, current_platform.Stream(), current_platform.Stream())
    try:
        compute = current_platform.Stream()
        x = torch.randn(4096, 4096, device=device)
        for value in range(1, 6):
            with current_platform.stream(compute):
                for _ in range(8):  # keep the compute stream busy first
                    x = x @ x
                    x = x / x.norm()
                gpu["a"][5].fill_(value)
                done = torch.Event()
                done.record(compute)
            events: list = []
            backend.launch_copy(
                [5], [4], is_store=True, event_idx=value, events_list=events, wait_event=done
            )
            _wait(events, value)
            assert torch.all(cpu["a"][4] == value)
    finally:
        backend.shutdown()


def test_pow2_chunks_cover_the_capacity_without_rounding_up():
    from vllm.v1.simple_kv_offload.worker import pow2_chunks

    gib = 1 << 30
    assert pow2_chunks(24 * gib) == [16 * gib, 8 * gib]
    assert pow2_chunks(25 * gib + gib // 2) == [16 * gib, 8 * gib, gib]
    assert pow2_chunks(16 * gib) == [16 * gib]
    assert pow2_chunks(gib // 2 + 5) == [gib // 2]
    assert pow2_chunks(0) == []


def test_roundtrip_across_cpu_segments():
    """A CPU cache split into pinned chunks: blocks on both sides of a chunk
    boundary land where the scheduler's block ids say."""
    device = torch.device("xpu", 0)
    gpu = {
        "a": torch.randint(-128, 127, (NUM_BLOCKS, 4096), dtype=torch.int8, device=device),
        "b": torch.randint(-128, 127, (NUM_BLOCKS, 1536), dtype=torch.int8, device=device),
    }
    sizes = [10, 6]  # CPU blocks 0-9 in the first chunk, 10-15 in the second
    cpu: dict = {"a": [], "b": []}
    for n in sizes:
        pool = torch.zeros(n * (4096 + 1536), dtype=torch.int8, pin_memory=True)
        cpu["a"].append(pool[: n * 4096].view(n, 4096))
        cpu["b"].append(pool[n * 4096 :].view(n, 1536))
    original = {k: v.clone() for k, v in gpu.items()}
    backend = XpuCopyBackend()
    backend.init(gpu, cpu, device, current_platform.Stream(), current_platform.Stream())
    try:
        stores: list = []
        cpu_ids = [8, 9, 10, 15]
        backend.launch_copy([1, 2, 3, 4], cpu_ids, is_store=True, event_idx=0, events_list=stores)
        _wait(stores, 0)
        for k in gpu:
            flat = torch.cat(cpu[k])
            assert torch.equal(flat[cpu_ids], original[k][[1, 2, 3, 4]].cpu())

        for k in gpu:
            gpu[k].zero_()
        torch.xpu.synchronize()
        loads: list = []
        backend.launch_copy(cpu_ids, [20, 21, 22, 23], is_store=False, event_idx=0, events_list=loads)
        _wait(loads, 0)
        for k in gpu:
            assert torch.equal(gpu[k][[20, 21, 22, 23]].cpu(), original[k][[1, 2, 3, 4]].cpu())
    finally:
        backend.shutdown()


@pytest.mark.parametrize("capacity_gib", [16, 24, 25.5, 0.75])
def test_scheduler_and_worker_agree_on_cpu_blocks(capacity_gib):
    """The scheduler's CPU block pool must not be larger than the blocks the
    worker pins, or a block id past the last chunk would be copied to."""
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
    )
    from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
    from vllm.v1.simple_kv_offload.worker import chunk_block_counts

    num_gpu_blocks = 1000
    per_block = [2_621_440, 2_621_440, 655_360, 917_504]  # uneven layers
    gpu_config = KVCacheConfig(
        num_blocks=num_gpu_blocks,
        kv_cache_tensors=[
            KVCacheTensor(size=b * num_gpu_blocks, shared_by=[f"l{i}"])
            for i, b in enumerate(per_block)
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"l{i}" for i in range(len(per_block))],
                FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=1, dtype=torch.float16),
            )
        ],
    )
    capacity = int(capacity_gib * (1 << 30))
    cpu_config = SimpleCPUOffloadScheduler._derive_cpu_config(gpu_config, capacity)
    assert cpu_config.num_blocks == sum(chunk_block_counts(capacity, sum(per_block)))
