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

import atexit
import os
import queue
import threading
import time
from dataclasses import dataclass

import torch

import vllm.envs as envs

_MAX_THREADS = 4


def worker_count() -> int:
    """Reader threads to run, from VLLM_MOE_DISK_IO_THREADS (default 2)."""
    return max(1, min(envs.VLLM_MOE_DISK_IO_THREADS, _MAX_THREADS))


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
        # The lock covers thread startup AND the enqueue, so a concurrent
        # shutdown() can never slip its sentinels between the two -- a
        # submitted job is always ahead of every sentinel in the queue and
        # therefore always drained.
        with self._lock:
            self._ensure_threads()
            self._jobs.put(_ReadJob(store, expert_id, dst, done, tag))

    def _ensure_threads(self) -> None:
        # Caller holds self._lock.
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

    def shutdown(self, timeout: float = 1.0) -> None:
        """Stop the reader threads; the pool restarts on the next submit().

        One sentinel per thread goes through the same FIFO the jobs use, so
        every already-submitted read completes and reports through its done
        queue before its thread exits. The join is bounded: a wedged read
        cannot hang the caller, and the threads are daemons regardless.
        """
        with self._lock:
            threads, self._threads = self._threads, []
            if self._pid != os.getpid():
                # A forked child never owns the parent's threads.
                return
            for _ in threads:
                self._jobs.put(None)
            self._pid = 0
        for t in threads:
            t.join(timeout)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
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


def shutdown_disk_load_worker(timeout: float = 1.0) -> None:
    """Stop the process-wide reader pool, if one was ever started.

    Registered at interpreter exit and safe to call at any time -- pending
    reads drain first (see DiskLoadWorker.shutdown) and the pool restarts
    lazily on the next read.
    """
    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
    if worker is not None:
        worker.shutdown(timeout)


atexit.register(shutdown_disk_load_worker)
