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
