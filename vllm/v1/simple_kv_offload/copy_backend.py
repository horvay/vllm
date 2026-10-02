# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DMA copy backend for GPU<->CPU block transfers."""

from __future__ import annotations

import queue
import threading

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
)

logger = init_logger(__name__)


class DmaCopyBackend:
    """cuMemcpyBatchAsync copy backend (background thread)."""

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream

        # Stores read the live KV cache -> STREAM (paired with the compute-done
        # wait in get_finished); loads read stable pinned host memory -> ANY.
        self._store_params = build_params(
            gpu_caches,
            cpu_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            cpu_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            )
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = q.get()
            if item is None:
                return
            (
                src_blocks,
                dst_blocks,
                params,
                is_store,
                event_idx,
                events_list,
                wait_event,
            ) = item
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            copy_blocks(src_blocks, dst_blocks, params)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))


# (first block of each segment, base address of each segment)
_Segments = tuple[np.ndarray, np.ndarray]


def _segments(cache: torch.Tensor | list[torch.Tensor]) -> tuple[_Segments, int]:
    parts = cache if isinstance(cache, list) else [cache]
    bpb = parts[0].stride(0) * parts[0].element_size()
    assert all(p.stride(0) * p.element_size() == bpb for p in parts)
    starts = np.cumsum([0] + [p.shape[0] for p in parts[:-1]]).astype(np.uint64)
    bases = np.array([p.data_ptr() for p in parts], dtype=np.uint64)
    return (starts, bases), bpb


def _block_ptrs(segments: _Segments, ids: np.ndarray, bpb: int) -> np.ndarray:
    starts, bases = segments
    seg = np.searchsorted(starts, ids, side="right") - 1
    return bases[seg] + (ids - starts[seg]) * np.uint64(bpb)


class XpuCopyBackend:
    """Intel XPU copy backend (background thread).

    XPU has no cuMemcpyBatchAsync; vLLM's ``swap_blocks_batch`` op drives the
    copy engine with one raw pointer copy per (layer, block), the path the
    OffloadingConnector already uses on XPU. Same interface as DmaCopyBackend,
    except that a cache may be a list of segments (consecutive runs of blocks
    in separate buffers), which the XPU worker uses for its pinned chunks.
    """

    def __init__(self) -> None:
        self._store_params: list[tuple[_Segments, _Segments, int]] | None = None
        self._load_params: list[tuple[_Segments, _Segments, int]] | None = None
        self._queue: queue.SimpleQueue | None = None
        self._thread: threading.Thread | None = None
        self._shutdown: bool = False

    @staticmethod
    def _params(
        src_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        dst_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    ) -> list[tuple[_Segments, _Segments, int]]:
        assert list(src_caches.keys()) == list(dst_caches.keys())
        layers = []
        for s, d in zip(src_caches.values(), dst_caches.values()):
            src, s_bpb = _segments(s)
            dst, d_bpb = _segments(d)
            assert s_bpb == d_bpb
            layers.append((src, dst, s_bpb))
        return layers

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        cpu_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        device: torch.device,
        load_stream: torch.Stream,
        store_stream: torch.Stream,
    ) -> None:
        self._store_params = self._params(gpu_caches, cpu_caches)
        self._load_params = self._params(cpu_caches, gpu_caches)
        self._queue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._copy_loop,
            args=(self._queue, device, load_stream, store_stream),
            daemon=True,
        )
        self._thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        params = self._store_params if is_store else self._load_params
        assert params is not None and self._queue is not None
        self._queue.put(
            (src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event)
        )

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        if self._queue is not None:
            self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    @staticmethod
    def _copy_loop(
        q: queue.SimpleQueue,
        device: torch.device,
        load_stream: torch.Stream,
        store_stream: torch.Stream,
    ) -> None:
        from vllm import _custom_ops as ops

        current_platform.set_device(device)
        # Pointer tables stay referenced until their copy's event completes.
        in_flight: list[tuple[torch.Event, tuple[torch.Tensor, ...]]] = []
        while True:
            item = q.get()
            if item is None:
                return
            src_blocks, dst_blocks, params, is_store, event_idx, events_list, wait_event = (
                item
            )
            in_flight = [entry for entry in in_flight if not entry[0].query()]
            stream = store_stream if is_store else load_stream
            if wait_event is not None:
                stream.wait_event(wait_event)
            n = len(src_blocks)
            tables: tuple[torch.Tensor, ...] = ()
            if n > 0:
                src_ids = np.array(src_blocks, dtype=np.uint64)
                dst_ids = np.array(dst_blocks, dtype=np.uint64)
                src_all = np.concatenate(
                    [_block_ptrs(src, src_ids, bpb) for src, _, bpb in params]
                )
                dst_all = np.concatenate(
                    [_block_ptrs(dst, dst_ids, bpb) for _, dst, bpb in params]
                )
                sizes = np.repeat(
                    np.array([bpb for _, _, bpb in params], dtype=np.uint64), n
                )
                tables = tuple(
                    torch.from_numpy(np.ascontiguousarray(a)).view(torch.uint64)
                    for a in (src_all, dst_all, sizes)
                )
                with current_platform.stream(stream):
                    ops.swap_blocks_batch(*tables)
            event = torch.Event()
            event.record(stream)
            in_flight.append((event, tables))
            events_list.append((event_idx, event))
