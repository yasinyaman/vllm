# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On-disk expert weight store for the MoE expert cache (prototype).

One file per MoE layer, one fixed-stride record per expert, each record
holding the runtime-layout tensors the cache serves (w13, w2, optional
per-expert scales) back to back, padded to the O_DIRECT alignment. Records
are read with O_DIRECT straight into pinned RAM-tier slots: the measured
NVMe ceiling is reached by single large aligned reads, and bypassing the
page cache keeps the RAM tier the only RAM this path uses.

The store is built once from the fully loaded weights and validated by a
JSON sidecar fingerprint on reuse. Building still requires the full weights
in memory -- intercepting the loading path so they never materialize is the
remaining (and larger) part of the disk tier, tracked in RFC #38256.
"""

import fcntl
import json
import os
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ALIGN = 4096

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float8_e4m3fn: "float8_e4m3fn",
}
_NAME_DTYPES = {v: k for k, v in _DTYPE_NAMES.items()}


def _as_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.contiguous().reshape(-1).view(torch.uint8).numpy())


@dataclass(frozen=True)
class _Field:
    name: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: torch.dtype


class DiskExpertStore:
    """Aligned per-expert records for one layer, readable with O_DIRECT."""

    def __init__(self, path: str, num_experts: int, fields: list[_Field]):
        self.path = path
        self.num_experts = num_experts
        self.fields = {f.name: f for f in fields}
        last = max(fields, key=lambda f: f.offset)
        raw = last.offset + last.nbytes
        self.record_stride = (raw + ALIGN - 1) // ALIGN * ALIGN
        self._fd: int | None = None
        self._o_direct = True
        self.is_complete = False
        self._wfd: int | None = None
        self._lock_f = None
        self._written: set[int] = set()
        self._identity: dict | None = None

    @staticmethod
    def _make_fields(specs: list[tuple[str, tuple[int, ...], torch.dtype]]):
        fields: list[_Field] = []
        offset = 0
        for name, shape, dtype in specs:
            numel = 1
            for d in shape:
                numel *= d
            per = numel * torch.empty(0, dtype=dtype).element_size()
            fields.append(_Field(name, offset, per, tuple(shape), dtype))
            offset += per
        return fields, offset

    @staticmethod
    def _fingerprint(
        num_experts: int, fields: list[_Field], identity: dict | None
    ) -> dict:
        return {
            "version": 2,
            "identity": identity or {},
            "num_experts": num_experts,
            "fields": [
                {
                    "name": f.name,
                    "offset": f.offset,
                    "nbytes": f.nbytes,
                    "shape": list(f.shape),
                    "dtype": _DTYPE_NAMES[f.dtype],
                }
                for f in fields
            ],
        }

    @classmethod
    def build(
        cls,
        path: str,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        identity: dict | None = None,
    ) -> "DiskExpertStore":
        """Create (or validate and reuse) the store for one layer.

        Tensors are indexed ``[num_experts, ...]`` and must already be in
        the exact layout the kernel consumes -- the record is a byte copy.

        ``identity`` names whose weights these are (model, revision, layer).
        It is part of the on-disk fingerprint: shapes and dtypes alone would
        let two different models of the same architecture silently reuse
        each other's store, which is the FP8-scale bug's failure mode all
        over again. Reuse requires an exact identity match.

        Build and validation run under an exclusive file lock, so concurrent
        engine processes pointed at the same directory serialize: one
        builds, the rest wait and reuse.
        """
        num_experts = w13.size(0)
        tensors: list[tuple[str, torch.Tensor]] = [("w13", w13), ("w2", w2)]
        if w13_scale is not None and w2_scale is not None:
            tensors += [("w13_scale", w13_scale), ("w2_scale", w2_scale)]

        fields, offset = cls._make_fields(
            [(n, tuple(t.shape[1:]), t.dtype) for n, t in tensors]
        )

        store = cls(path, num_experts, fields)
        sidecar = path + ".json"
        want = cls._fingerprint(num_experts, fields, identity)

        with open(path + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)

            if os.path.exists(sidecar) and os.path.exists(path):
                with open(sidecar) as f:
                    have = json.load(f)
                if have == want:
                    logger.info("DiskExpertStore: reusing %s", path)
                    store.is_complete = True
                    return store
                logger.warning(
                    "DiskExpertStore: fingerprint mismatch, rebuilding %s", path
                )

            cpu = [(n, t.detach().cpu()) for n, t in tensors]
            pad = b"\x00" * (store.record_stride - offset)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                for e in range(num_experts):
                    for _, t in cpu:
                        f.write(_as_bytes(t[e]))
                    if pad:
                        f.write(pad)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            with open(sidecar, "w") as f:
                json.dump(want, f)
            logger.info(
                "DiskExpertStore: wrote %s (%d experts x %.1f MiB)",
                path,
                num_experts,
                store.record_stride / 2**20,
            )
            store.is_complete = True
            return store

    def _open(self) -> int:
        if self._fd is None:
            flags = os.O_RDONLY
            o_direct = getattr(os, "O_DIRECT", 0)
            try:
                self._fd = os.open(self.path, flags | o_direct)
                self._o_direct = bool(o_direct)
            except OSError:
                self._fd = os.open(self.path, flags)
                self._o_direct = False
            if not self._o_direct:
                logger.warning_once(
                    "DiskExpertStore: O_DIRECT unavailable for %s; reads go "
                    "through the page cache and RAM accounting is off.",
                    self.path,
                )
        return self._fd

    def read_record(self, expert_id: int, dst: torch.Tensor) -> int:
        """Read one expert's full record into ``dst``.

        ``dst`` is a pinned uint8 tensor of ``record_stride`` bytes whose
        start address is ALIGN-aligned (any row of a pinned
        ``[slots, record_stride]`` pool qualifies: cudaHostAlloc returns
        page-aligned memory and the stride is a multiple of ALIGN).
        """
        assert dst.dtype == torch.uint8 and dst.numel() == self.record_stride
        fd = self._open()
        view = memoryview(dst.numpy())
        offset = expert_id * self.record_stride
        got = 0
        while got < self.record_stride:
            try:
                n = os.preadv(fd, [view[got:]], offset + got)
            except OSError as e:
                raise OSError(
                    e.errno,
                    f"DiskExpertStore preadv failed: errno={e.errno} "
                    f"path={self.path} expert={expert_id} got={got} "
                    f"ptr={dst.data_ptr()} ptr%4096={dst.data_ptr() % 4096} "
                    f"off={offset + got} off%4096={(offset + got) % 4096} "
                    f"len={self.record_stride - got} "
                    f"len%4096={(self.record_stride - got) % 4096} "
                    f"o_direct={self._o_direct}",
                ) from e
            if n <= 0:
                raise OSError(
                    f"DiskExpertStore: short read at expert {expert_id} "
                    f"({got}/{self.record_stride} bytes)"
                )
            got += n
        return got

    def field_view(self, pool_row: torch.Tensor, name: str) -> torch.Tensor:
        """Typed view of one field inside a record-sized uint8 row."""
        f = self.fields[name]
        flat = pool_row[f.offset : f.offset + f.nbytes]
        return flat.view(f.dtype).reshape(f.shape)

    @classmethod
    def create_for_streaming(
        cls,
        path: str,
        num_experts: int,
        specs: list[tuple[str, tuple[int, ...], torch.dtype]],
        identity: dict | None = None,
    ) -> "DiskExpertStore":
        """Open a store that the weight loader fills record by record.

        This is the loading-path interception: shapes and dtypes are known
        before any weight arrives, so the store can be sized up front and
        each expert written the moment its shards complete -- the full
        ``[num_experts, ...]`` tensor never exists.

        If a store with a matching fingerprint already exists, it is reused
        and ``is_complete`` is True from the start; the loader then skips
        expert writes entirely. The exclusive lock is held until
        ``finalize()`` so concurrent processes serialize on the whole load.
        """
        fields, _ = cls._make_fields(specs)
        store = cls(path, num_experts, fields)
        store._identity = identity

        # Held across the whole streaming load, released in finalize() --
        # a context manager cannot express that lifetime.
        store._lock_f = open(path + ".lock", "w")  # noqa: SIM115
        fcntl.flock(store._lock_f, fcntl.LOCK_EX)

        sidecar = path + ".json"
        want = cls._fingerprint(num_experts, fields, identity)
        if os.path.exists(sidecar) and os.path.exists(path):
            with open(sidecar) as f:
                if json.load(f) == want:
                    logger.info("DiskExpertStore: streaming reuse of %s", path)
                    store.is_complete = True
                    store._release_lock()
                    return store
            logger.warning(
                "DiskExpertStore: fingerprint mismatch, restreaming %s", path
            )

        store._wfd = os.open(path + ".tmp", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(store._wfd, num_experts * store.record_stride)
        return store

    def write_record(self, expert_id: int, src: torch.Tensor) -> None:
        """Write one completed expert record during a streaming load."""
        assert self._wfd is not None and not self.is_complete
        assert src.dtype == torch.uint8 and src.numel() == self.record_stride
        os.pwrite(self._wfd, bytes(src.numpy()), expert_id * self.record_stride)
        self._written.add(expert_id)

    def finalize(self) -> None:
        """Seal a streaming store: every expert written, sidecar published."""
        if self.is_complete:
            return
        assert self._wfd is not None
        missing = self.num_experts - len(self._written)
        if missing:
            raise RuntimeError(
                f"DiskExpertStore: streaming load ended with {missing} of "
                f"{self.num_experts} experts unwritten for {self.path}"
            )
        os.fsync(self._wfd)
        os.close(self._wfd)
        self._wfd = None
        os.replace(self.path + ".tmp", self.path)
        with open(self.path + ".json", "w") as f:
            json.dump(
                self._fingerprint(
                    self.num_experts, list(self.fields.values()), self._identity
                ),
                f,
            )
        self.is_complete = True
        self._release_lock()
        logger.info(
            "DiskExpertStore: streamed %s (%d experts x %.1f MiB)",
            self.path,
            self.num_experts,
            self.record_stride / 2**20,
        )

    def _release_lock(self) -> None:
        if self._lock_f is not None:
            fcntl.flock(self._lock_f, fcntl.LOCK_UN)
            self._lock_f.close()
            self._lock_f = None

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._wfd is not None:
            os.close(self._wfd)
            self._wfd = None
        self._release_lock()
