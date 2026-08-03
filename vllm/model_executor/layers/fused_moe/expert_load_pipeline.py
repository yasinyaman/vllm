# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background disk reads for the MoE expert cache.

The reader pool is CUDA-free by design: it touches pinned host tensors and
file descriptors, nothing else. All CUDA interaction -- the slot-event
synchronize before a read is submitted, the H2D issue after it completes --
stays on the submitting thread, which keeps the pipeline's correctness
argument the same as the serial path's: queue handoff orders the bytes
before the H2D, stream order orders the H2D before the kernel.

Thread count: the measured NVMe reaches its single-stream ceiling only at
~22 MiB blocks; expert-sized records (9-12 MiB) read at ~5.4 of 6.9 GB/s on
one thread, and a queue depth of 2 recovers most of the difference. Fan-out
hurts instead of helping -- 16 concurrent readers pushed p99 read latency
from 4 ms to 128 ms on the same device -- so the count is clamped hard.
"""

import os
import queue
import threading
import time
from dataclasses import dataclass

import torch

_MAX_THREADS = 4


def worker_count() -> int:
    """Reader threads to run, from VLLM_MOE_DISK_IO_THREADS (default 2)."""
    n = int(os.environ.get("VLLM_MOE_DISK_IO_THREADS", "2"))
    return max(1, min(n, _MAX_THREADS))


@dataclass
class _ReadJob:
    store: object  # duck-typed DiskExpertStore
    expert_id: int
    dst: torch.Tensor  # pinned destination row, exclusively this job's
    done: queue.SimpleQueue  # receives (tag, exc | None, seconds)
    tag: int


class DiskLoadWorker:
    """Process-wide pool of reader threads for expert records.

    Submission is bounded by construction -- callers submit at most one
    prepare() plan's reads (<= GPU capacity) and drain them before
    returning -- so the queue cannot grow without bound. A job can never
    kill a thread: every outcome, including an exception, is reported
    through the job's ``done`` queue.
    """

    def __init__(self) -> None:
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._pid = 0

    def submit(
        self,
        store,
        expert_id: int,
        dst: torch.Tensor,
        done: queue.SimpleQueue,
        tag: int,
    ) -> None:
        self._ensure_threads()
        self._jobs.put(_ReadJob(store, expert_id, dst, done, tag))

    def _ensure_threads(self) -> None:
        if self._pid == os.getpid() and self._threads:
            return
        with self._lock:
            if self._pid == os.getpid() and self._threads:
                return
            if self._pid not in (0, os.getpid()):
                # Forked child: the parent's threads do not exist here, and
                # whatever sat in the parent's queue belongs to the parent.
                self._jobs = queue.SimpleQueue()
            self._threads = [
                threading.Thread(target=self._run, name=f"moe-disk-io-{i}", daemon=True)
                for i in range(worker_count())
            ]
            for t in self._threads:
                t.start()
            self._pid = os.getpid()

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            t0 = time.perf_counter()
            try:
                job.store.read_record(job.expert_id, job.dst)
            except BaseException as e:
                job.done.put((job.tag, e, time.perf_counter() - t0))
            else:
                job.done.put((job.tag, None, time.perf_counter() - t0))


_worker: DiskLoadWorker | None = None
_worker_lock = threading.Lock()


def get_disk_load_worker() -> DiskLoadWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = DiskLoadWorker()
    return _worker
