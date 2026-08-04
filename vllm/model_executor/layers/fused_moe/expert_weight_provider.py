# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.expert_disk_store import DiskExpertStore
from vllm.model_executor.layers.fused_moe.expert_load_pipeline import (
    get_disk_load_worker,
)

logger = init_logger(__name__)

_STATS_LOG_INTERVAL = 1000


class _PinnedRegion:
    """Exactly-sized, page-aligned, page-locked host buffer.

    ``Tensor.pin_memory()`` lands in torch's caching host allocator, which
    buckets requests to powers of two -- measured on GB10, a 604 MB
    per-layer pool request occupies 1027 MB of RSS, inflating the disk
    tier's memory budget by 40-70% across a model's layers. Allocating
    pageable memory and page-locking it in place with cudaHostRegister
    costs exactly what was asked (604 -> 604 MB), torch still reports the
    region pinned (async H2D stays async; 55.7 vs 54.7 GB/s measured) and
    O_DIRECT reads into it work unchanged. Falls back to pin_memory() if
    registration is unavailable.
    """

    def __init__(self, nbytes: int) -> None:
        from vllm.model_executor.layers.fused_moe.expert_disk_store import ALIGN

        self.registered = False
        self._reg_ptr = 0
        backing = torch.empty(nbytes + ALIGN, dtype=torch.uint8)
        skip = (-backing.data_ptr()) % ALIGN
        region = backing[skip : skip + nbytes]
        ok = False
        if torch.cuda.is_available():
            try:
                ret = torch.cuda.cudart().cudaHostRegister(region.data_ptr(), nbytes, 0)
                ok = int(ret) == 0
            except Exception:
                ok = False
            if ok:
                self.registered = True
                self._reg_ptr = region.data_ptr()
            else:
                logger.warning_once(
                    "cudaHostRegister unavailable; RAM pool falls back to "
                    "the caching allocator's pin_memory (bucketed sizes)."
                )
                backing = torch.empty(nbytes + ALIGN, dtype=torch.uint8).pin_memory()
                skip = (-backing.data_ptr()) % ALIGN
                region = backing[skip : skip + nbytes]
        self._backing = backing
        self.tensor = region

    def __del__(self) -> None:
        if getattr(self, "registered", False):
            try:
                torch.cuda.cudart().cudaHostUnregister(self._reg_ptr)
            except Exception:
                pass


def _pinned_cpu_copy(src: torch.Tensor) -> torch.Tensor:
    """Pinned CPU copy of *src* with exactly one cross-device transfer.

    ``src.cpu().pin_memory()`` stages GPU tensors through pageable memory,
    copying twice; copying straight into a pinned allocation halves init
    time and transient host RAM for multi-GB expert tensors.
    """
    if src.device.type == "cpu":
        return src if src.is_pinned() else src.pin_memory()
    dst = torch.empty_like(src, device="cpu", pin_memory=True)
    dst.copy_(src)
    return dst


# How a forward too wide for the cache is broken up.
#   "token"  -- split the rows; every token's full sum still happens inside one
#               kernel call, so output matches the uncached path exactly. Costs
#               one launch per chunk, which approaches one per token as
#               capacity approaches top_k.
#   "expert" -- split the experts; one launch per ceil(experts/capacity)
#               regardless of batch size, and each expert is fetched at most
#               once per forward. Each group's partial sum is rounded to the
#               model dtype before being accumulated, so results differ from
#               the uncached path at rounding level.
MoECacheSplit = Literal["token", "expert"]

# Every row: what the expert split passes, since it never cuts the batch.
_ALL_ROWS = slice(None)


@dataclass
class ExpertWeightResult:
    """GPU-resident expert weights ready for kernel consumption.

    ``expert_map`` follows the expert-parallel convention the kernels already
    understand: global expert id to buffer slot, or -1 for experts that are not
    resident. Pairs routed to -1 are dropped during alignment, which is what
    lets one forward be evaluated a group of experts at a time.
    """

    w1: torch.Tensor
    w2: torch.Tensor
    expert_map: torch.Tensor
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None


@dataclass
class _LoadOp:
    """One planned cache fill: which expert, and where its bytes land.

    Produced by ``_plan_group`` at decision time, consumed by
    ``_execute_plan``. ``ram_slot`` is -1 in full-DRAM mode. ``needs_read``
    marks RAM misses whose bytes must come from disk before the H2D. The
    two progress flags are what rollback reads to tell garbage from real
    bytes: ``read_done`` means the RAM slot's bytes are valid,
    ``h2d_issued`` means the op completed and its claims must survive.
    """

    expert_id: int
    gpu_slot: int
    ram_slot: int = -1
    needs_read: bool = False
    read_done: bool = False
    h2d_issued: bool = False


class CachedWeightProvider:
    """GPU LRU cache backed by CPU pinned memory.

    Keeps capacity expert weight tensors in a fixed-size GPU scratch
    buffer. All expert weights reside in CPU pinned memory; only the N
    hottest experts are mirrored into the GPU buffer.

    Uses LFRU (frequency-weighted LRU) eviction: score = freq / age.
    This prevents early layers from monopolizing the cache — a known
    problem with pure LRU in sequential MoE execution where early
    layers always appear "recently used."

    prepare() copies any missing experts from CPU to GPU, evicting the
    lowest-scored resident entry when the buffer is full, and returns an
    ExpertWeightResult whose expert_map selects them. Forwards needing more
    experts than fit are handled by run_with_expert_cache(), which splits
    them according to ``split``.
    """

    def __init__(
        self,
        capacity: int,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        split: MoECacheSplit = "token",
        ram_capacity: int = 0,
        disk_store: DiskExpertStore | None = None,
    ) -> None:
        num_experts = w13_weight.size(0)
        if disk_store is not None:
            # Streaming load hands in zero-expert placeholders; the store is
            # the authority on the expert count.
            num_experts = disk_store.num_experts

        self.capacity = capacity
        self.split: MoECacheSplit = split
        self._num_experts = num_experts
        self.hits = 0
        self.misses = 0
        self.ram_hits = 0
        self.ram_misses = 0
        self._prepare_calls = 0

        # Wall-time attribution for the disk tier, in seconds/bytes.
        # Decode through the disk tier is host-serialized in prepare(), so
        # host wall time is the frame that explains tok/s. Incremented only
        # on disk-path code; the DRAM and cache-off paths never touch them.
        # bench/disk_tier_bench.py reads these to split the gap between NVMe
        # bandwidth inside preadv and what prepare() achieves end to end.
        self.t_prepare = 0.0
        self.t_disk_read = 0.0
        self.n_disk_bytes = 0
        self.t_event_wait = 0.0
        self.t_h2d_issue = 0.0
        self.t_mapping = 0.0
        self.max_read_s = 0.0

        if w13_weight.device.type == "cpu":
            cuda_device = torch.accelerator.current_accelerator()
        else:
            cuda_device = w13_weight.device

        # Three-tier mode: experts live on disk, a pinned RAM pool of
        # ram_capacity records is the warm tier, and the full-weight pinned
        # copies below are skipped entirely -- that is the point.
        self._disk_store = disk_store
        # VLLM_MOE_DISK_PIPELINE=0 keeps disk reads on the calling thread,
        # bit-identical to the pre-pipeline code. Only meaningful with a
        # disk store; the DRAM path never reads.
        self._pipeline = disk_store is not None and envs.VLLM_MOE_DISK_PIPELINE
        if disk_store is not None:
            if ram_capacity < capacity:
                raise ValueError(
                    f"ram_capacity ({ram_capacity}) must be >= the GPU "
                    f"capacity ({capacity}); one prepare() call can pin up "
                    f"to that many experts in RAM at once."
                )
            self.ram_capacity = ram_capacity
            stride = disk_store.record_stride
            # _PinnedRegion gives a page-aligned start (O_DIRECT preadv into
            # a misaligned row fails with EINVAL) and exact-size pinning;
            # rows stay aligned because the stride is a multiple of ALIGN.
            from vllm.model_executor.layers.fused_moe.expert_disk_store import (
                ALIGN,
            )

            self._ram_pool_region = _PinnedRegion(ram_capacity * stride)
            self._ram_pool: torch.Tensor | None = self._ram_pool_region.tensor.view(
                ram_capacity, stride
            )
            assert self._ram_pool.data_ptr() % ALIGN == 0
            self._ram_w13 = [
                disk_store.field_view(self._ram_pool[i], "w13")
                for i in range(ram_capacity)
            ]
            self._ram_w2 = [
                disk_store.field_view(self._ram_pool[i], "w2")
                for i in range(ram_capacity)
            ]
            has_scales = "w13_scale" in disk_store.fields
            self._ram_w13_scale = (
                [
                    disk_store.field_view(self._ram_pool[i], "w13_scale")
                    for i in range(ram_capacity)
                ]
                if has_scales
                else None
            )
            self._ram_w2_scale = (
                [
                    disk_store.field_view(self._ram_pool[i], "w2_scale")
                    for i in range(ram_capacity)
                ]
                if has_scales
                else None
            )
            # A RAM slot is both a disk-read destination and an async-H2D
            # source. Before a slot is overwritten by a new disk read, any
            # H2D still in flight from it must have finished, or the GPU
            # copy could observe the new expert's bytes. One event per slot,
            # recorded after each H2D from that slot, waited on before reuse.
            self._ram_events = (
                [torch.cuda.Event() for _ in range(ram_capacity)]
                if torch.cuda.is_available()
                else None
            )
            self._ram_lru: dict[int, list] = {}
            self._ram_clock = 0
            self._ram_free = list(range(ram_capacity))
            self._cpu_w13 = None
            self._cpu_w2 = None
        else:
            self.ram_capacity = 0
            self._ram_pool = None
            self._cpu_w13 = _pinned_cpu_copy(w13_weight)
            self._cpu_w2 = _pinned_cpu_copy(w2_weight)

        self._buf_w13: torch.Tensor = torch.empty(
            capacity,
            *w13_weight.shape[1:],
            dtype=w13_weight.dtype,
            device=cuda_device,
        )
        self._buf_w2: torch.Tensor = torch.empty(
            capacity,
            *w2_weight.shape[1:],
            dtype=w2_weight.dtype,
            device=cuda_device,
        )

        if w13_scale is not None and w2_scale is not None:
            # Pinned for the same reason the weights are: these are copied on
            # every miss, and pageable source memory forces a staging copy.
            # In three-tier mode the RAM pool is the pinned source instead.
            self._cpu_w13_scale: torch.Tensor | None = (
                None if disk_store is not None else _pinned_cpu_copy(w13_scale)
            )
            self._cpu_w2_scale: torch.Tensor | None = (
                None if disk_store is not None else _pinned_cpu_copy(w2_scale)
            )
            self._buf_w13_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w13_scale.shape[1:],
                dtype=w13_scale.dtype,
                device=cuda_device,
            )
            self._buf_w2_scale: torch.Tensor | None = torch.empty(
                capacity,
                *w2_scale.shape[1:],
                dtype=w2_scale.dtype,
                device=cuda_device,
            )
        else:
            self._cpu_w13_scale = None
            self._cpu_w2_scale = None
            self._buf_w13_scale = None
            self._buf_w2_scale = None

        # LFRU state: {expert_id: [slot, freq, last_access_clock]}
        # Eviction score = freq / (clock - last_access + 1). Lower = evict first.
        self._lru: dict[int, list] = {}
        self._clock: int = 0
        self._free_slots: list[int] = list(range(capacity))

        # Expert map handed to the kernel: expert id to slot for the group
        # being evaluated, -1 for everything else. Rebuilt each prepare() --
        # it must expose exactly the requested group, not whatever else
        # happens to still be resident, or experts already summed in an
        # earlier group would be counted twice. Residency itself lives in
        # _lru; this is only the view the kernel gets. Built in a pinned host
        # mirror and uploaded once, rather than a transfer per entry.
        self._mapping: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=cuda_device
        )
        self._mapping_host: torch.Tensor = torch.full(
            (num_experts,), -1, dtype=torch.int32
        ).pin_memory()

    @property
    def buf_w13(self) -> torch.Tensor:
        return self._buf_w13

    @property
    def buf_w2(self) -> torch.Tensor:
        return self._buf_w2

    @property
    def buf_w13_scale(self) -> torch.Tensor | None:
        return self._buf_w13_scale

    @property
    def buf_w2_scale(self) -> torch.Tensor | None:
        return self._buf_w2_scale

    def invalidate(self, expert_id: int) -> None:
        """Remove *expert_id* from the cache, returning its slot to the free
        list.  No-op if the expert is not currently cached."""
        if expert_id in self._lru:
            entry = self._lru.pop(expert_id)
            self._free_slots.append(entry[0])

    def _plan_ram_slot(self, expert_id: int, needed: set[int]) -> tuple[int, bool]:
        """Pick the RAM slot for *expert_id*, evicting at decision time.

        Mirrors the GPU tier's LFRU, including the rule that experts the
        current call still needs are never evicted -- ram_capacity >= the GPU
        capacity guarantees a victim outside ``needed`` exists.

        On a miss the slot is claimed in ``_ram_lru`` before its bytes exist;
        ``_execute_plan`` fills them before anything observes the slot.
        Claiming at plan time is also what keeps the slot from being handed
        out twice in one call: it now holds a ``needed`` expert, which the
        victim scan skips.

        Returns:
            ``(slot, needs_read)`` -- ``needs_read`` is True on a RAM miss,
            whose disk read ``_execute_plan`` still owes.
        """
        entry = self._ram_lru.get(expert_id)
        if entry is not None:
            self._ram_clock += 1
            entry[1] += 1
            entry[2] = self._ram_clock
            self.ram_hits += 1
            return entry[0], False

        if self._ram_free:
            slot = self._ram_free.pop()
        else:
            best_key = None
            best_score = float("inf")
            for k, (s, freq, last) in self._ram_lru.items():
                if k in needed:
                    continue
                score = freq / (self._ram_clock - last + 1)
                if score < best_score:
                    best_score = score
                    best_key = k
            assert best_key is not None
            slot = self._ram_lru.pop(best_key)[0]

        self._ram_clock += 1
        self._ram_lru[expert_id] = [slot, 1, self._ram_clock]
        self.ram_misses += 1
        return slot, True

    def _read_into_ram_slot(self, expert_id: int, ram_slot: int) -> None:
        """Fill *ram_slot* from disk, waiting out any H2D still using it.

        The slot event is the other half of the ``_ram_events`` protocol
        declared at their allocation: it must have passed before the slot's
        bytes are overwritten, or an in-flight H2D from this slot could
        observe the new expert.
        """
        if self._ram_events is not None:
            t0 = time.perf_counter()
            self._ram_events[ram_slot].synchronize()
            self.t_event_wait += time.perf_counter() - t0
        assert self._disk_store is not None and self._ram_pool is not None
        t0 = time.perf_counter()
        self._disk_store.read_record(expert_id, self._ram_pool[ram_slot])
        dt = time.perf_counter() - t0
        self.t_disk_read += dt
        self.n_disk_bytes += self._disk_store.record_stride
        if dt > self.max_read_s:
            self.max_read_s = dt

    def _plan_group(self, unique_ids: list[int]) -> list[_LoadOp]:
        """Make every LFRU decision for one prepare() call; no bytes move.

        Experts requested here must never be evicted to make room for
        another one in the same call -- their slot would be handed to a
        different expert while they are still expected to be resident. A
        freshly loaded expert has freq=1, exactly the lowest LFRU score, so
        it is the first eviction candidate. The map built at the end of
        prepare() reads every requested expert back out of _lru, so
        violating this raises rather than corrupting silently, but it must
        not happen at all.

        State is mutated at decision time, in request order, exactly as the
        pre-plan inline loop did, so hit/miss counts, clocks and eviction
        choices are bit-identical to it. An entry can therefore be resident
        in state while its bytes are still pending in the returned plan;
        nothing observes the bytes before ``_execute_plan`` returns, the
        exclusion above keeps a pending slot from being re-assigned within
        the call, and ``_execute_plan``'s rollback keeps a mid-plan failure
        from leaving a poisoned entry behind.
        """
        needed = set(unique_ids)
        ops: list[_LoadOp] = []
        for expert_id in unique_ids:
            if expert_id in self._lru:
                # Cache hit: update frequency and recency
                self._clock += 1
                entry = self._lru[expert_id]
                entry[1] += 1  # freq
                entry[2] = self._clock  # last access
                self.hits += 1
                continue

            # Cache miss: need to load expert
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                # Evict entry with lowest freq/age score
                best_key = None
                best_score = float("inf")
                for k, (s, freq, last) in self._lru.items():
                    if k in needed:
                        continue
                    age = self._clock - last + 1
                    score = freq / age
                    if score < best_score:
                        best_score = score
                        best_key = k
                # len(unique_ids) <= capacity is enforced by prepare(), so at
                # least one cached expert is outside `needed` whenever the
                # buffer is full and a miss remains to be served.
                assert best_key is not None
                slot = self._lru.pop(best_key)[0]

            if self._disk_store is not None:
                ram_slot, needs_read = self._plan_ram_slot(expert_id, needed)
            else:
                ram_slot, needs_read = -1, False
            self._clock += 1
            self._lru[expert_id] = [slot, 1, self._clock]
            self.misses += 1
            ops.append(_LoadOp(expert_id, slot, ram_slot, needs_read))
        return ops

    def _issue_h2d(self, op: _LoadOp) -> None:
        """Enqueue one op's H2D copies on the current stream.

        Copy expert weights into the GPU slot -- from the RAM tier
        (disk-backed) or from the full pinned copies. Recording the slot
        event after the copies is the first half of the ``_ram_events``
        protocol; ``_read_into_ram_slot`` waits on it before reuse.
        """
        slot = op.gpu_slot
        if self._disk_store is not None:
            rslot = op.ram_slot
            t0 = time.perf_counter()
            self._buf_w13[slot].copy_(self._ram_w13[rslot], non_blocking=True)
            self._buf_w2[slot].copy_(self._ram_w2[rslot], non_blocking=True)
            if self._buf_w13_scale is not None:
                assert self._ram_w13_scale is not None
                assert self._ram_w2_scale is not None
                assert self._buf_w2_scale is not None
                self._buf_w13_scale[slot].copy_(
                    self._ram_w13_scale[rslot], non_blocking=True
                )
                self._buf_w2_scale[slot].copy_(
                    self._ram_w2_scale[rslot], non_blocking=True
                )
            if self._ram_events is not None:
                self._ram_events[rslot].record()
            self.t_h2d_issue += time.perf_counter() - t0
        else:
            expert_id = op.expert_id
            assert self._cpu_w13 is not None
            assert self._cpu_w2 is not None
            self._buf_w13[slot].copy_(self._cpu_w13[expert_id], non_blocking=True)
            self._buf_w2[slot].copy_(self._cpu_w2[expert_id], non_blocking=True)
            if self._buf_w13_scale is not None:
                assert self._cpu_w13_scale is not None
                assert self._cpu_w2_scale is not None
                assert self._buf_w2_scale is not None
                self._buf_w13_scale[slot].copy_(
                    self._cpu_w13_scale[expert_id], non_blocking=True
                )
                self._buf_w2_scale[slot].copy_(
                    self._cpu_w2_scale[expert_id], non_blocking=True
                )
        op.h2d_issued = True

    def _execute_plan(self, ops: list[_LoadOp]) -> None:
        """Move the planned bytes.

        A failed read must never leave ``_lru``/``_ram_lru`` claiming an
        expert whose bytes were not read -- a later prepare() would "hit"
        garbage and the kernel would silently compute with it. On error,
        every claim whose bytes did not fully arrive is rolled back before
        re-raising: the forward fails loudly and the cache stays consistent
        for whatever retries. A RAM entry is kept when its read finished
        (the bytes are real, only the H2D was lost with the failing
        forward); it is dropped when the read never ran or died halfway,
        either of which leaves garbage.
        """
        if self._pipeline and any(op.needs_read for op in ops):
            self._execute_plan_pipelined(ops)
            return
        try:
            for op in ops:
                if op.needs_read:
                    self._read_into_ram_slot(op.expert_id, op.ram_slot)
                    op.read_done = True
                self._issue_h2d(op)
        except BaseException:
            self._rollback_unfinished(ops)
            raise

    def _execute_plan_pipelined(self, ops: list[_LoadOp]) -> None:
        """Run the plan's disk reads on the reader pool.

        Happens-before chain for a RAM slot S feeding a GPU slot G:
          1. The plan assigned S exclusively to one expert for this call
             (after assignment S holds a ``needed`` expert, which every
             victim scan skips -- no slot is handed out twice).
          2. This thread waits out S's slot event: every H2D still reading
             S has finished on the GPU before S is given to a reader. All
             event waits come first, so the submission side makes no CUDA
             call after anything is in flight.
          3. A reader thread fills S; its completion message through the
             queue is the CPU-side ordering that publishes the bytes.
          4. This thread issues the S -> G copy on the current stream and
             records S's event behind it.
          5. The kernel launches later on the same stream, ordered after
             every H2D -- the same guarantee the serial path relies on.

        With more than one reader, completions arrive out of order, so ops
        are keyed by tag. Every submitted read is drained even after a
        failure -- a reader must never be left writing into a slot whose
        claim was already rolled back.
        """
        done: queue.SimpleQueue = queue.SimpleQueue()
        worker = get_disk_load_worker()
        pending: dict[int, _LoadOp] = {}
        first_exc: BaseException | None = None
        try:
            if self._ram_events is not None:
                for op in ops:
                    if op.needs_read:
                        t0 = time.perf_counter()
                        self._ram_events[op.ram_slot].synchronize()
                        self.t_event_wait += time.perf_counter() - t0
            for op in ops:
                if not op.needs_read:
                    # Warm bytes: overlap these H2Ds with the reads below.
                    self._issue_h2d(op)
            assert self._disk_store is not None and self._ram_pool is not None
            for tag, op in enumerate(ops):
                if op.needs_read:
                    worker.submit(
                        self._disk_store,
                        op.expert_id,
                        self._ram_pool[op.ram_slot],
                        done,
                        tag,
                    )
                    pending[tag] = op
        except BaseException as e:
            first_exc = e

        stride = self._disk_store.record_stride if self._disk_store else 0
        try:
            while pending:
                tag, exc, dt = done.get()
                op = pending.pop(tag)
                if exc is not None:
                    if first_exc is None:
                        first_exc = exc
                    continue
                op.read_done = True
                self.t_disk_read += dt
                self.n_disk_bytes += stride
                if dt > self.max_read_s:
                    self.max_read_s = dt
                if first_exc is None:
                    try:
                        self._issue_h2d(op)
                    except BaseException as e:
                        first_exc = e
        except BaseException:
            # An interrupt landing in done.get() -- KeyboardInterrupt,
            # chiefly, since the drain is where this thread blocks -- must
            # not skip the cleanup: submitted reads are still drained so no
            # reader is left writing into a slot whose claim is being rolled
            # back, and unfinished claims must not survive to serve garbage
            # to a later prepare(). Reads that did land keep their RAM
            # entries, same as every other failure path.
            while pending:
                tag, exc, _ = done.get()
                op = pending.pop(tag)
                if exc is None:
                    op.read_done = True
            self._rollback_unfinished(ops)
            raise
        if first_exc is not None:
            self._rollback_unfinished(ops)
            raise first_exc

    def _rollback_unfinished(self, ops: list[_LoadOp]) -> None:
        """Undo the plan-time claims of every op that did not complete.

        Completed ops (``h2d_issued``) keep both claims. An op whose read
        finished keeps its RAM entry -- the bytes are real, only the H2D
        was lost with the failing forward -- but never its GPU claim. An
        op whose read never ran or died halfway loses both.
        """
        for op in ops:
            if op.h2d_issued:
                continue
            entry = self._lru.pop(op.expert_id, None)
            if entry is not None:
                self._free_slots.append(entry[0])
            if op.needs_read and not op.read_done:
                rentry = self._ram_lru.pop(op.expert_id, None)
                if rentry is not None:
                    self._ram_free.append(rentry[0])

    @torch.compiler.disable
    def plan_chunks(self, topk_ids: torch.Tensor) -> list[tuple[slice, list[int]]]:
        """Row ranges whose combined unique expert count fits the cache.

        Routing is per token, so evaluating a subset of rows and concatenating
        the results is equivalent to evaluating the whole batch -- and every
        token's sum still happens in a single kernel call, which is why this
        split reproduces the uncached output exactly.

        Returns a single full-width slice when the batch already fits, which is
        the common case, so callers pay nothing extra for it. Each slice comes
        with the expert ids it needs, computed from host data this method
        already has, so ``prepare()`` need not synchronize again per chunk.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            ``(row slice, unique expert ids)`` pairs covering ``topk_ids``.

        Raises:
            RuntimeError: if one token alone routes to more experts than the
                cache can hold, which no amount of splitting can fix.
        """
        num_rows = topk_ids.size(0)
        if num_rows == 0:
            return [(slice(0, 0), [])]

        # The common case only needs the distinct ids, which the device can
        # reduce far more cheaply than transferring every row.
        unique = [e for e in topk_ids.unique().tolist() if e >= 0]
        if len(unique) <= self.capacity:
            return [(slice(0, num_rows), unique)]

        rows = topk_ids.tolist()
        chunks: list[tuple[slice, list[int]]] = []
        start = 0
        seen: set[int] = set()
        for i, row in enumerate(rows):
            row_ids = {e for e in row if e >= 0}
            if len(row_ids) > self.capacity:
                raise RuntimeError(
                    f"CachedWeightProvider: one token routes to "
                    f"{len(row_ids)} experts but "
                    f"--moe-expert-cache-size={self.capacity}. "
                    f"Set --moe-expert-cache-size >= {len(row_ids)}."
                )
            if len(seen | row_ids) > self.capacity:
                chunks.append((slice(start, i), sorted(seen)))
                start = i
                seen = row_ids
            else:
                seen |= row_ids
        chunks.append((slice(start, num_rows), sorted(seen)))
        return chunks

    @torch.compiler.disable
    def plan_expert_groups(self, topk_ids: torch.Tensor) -> list[list[int]]:
        """Split the forward's experts into groups the cache can hold at once.

        A token's output is the weighted sum of its experts' outputs, so the
        sum can be taken a few experts at a time and accumulated: run the
        kernel once per group with an ``expert_map`` that hides the others,
        then add the results. Every (token, expert) pair is still computed
        exactly once, in whichever group owns its expert.

        Cost is ``ceil(experts_used / capacity)`` kernel launches, independent
        of how many tokens are in the batch, and each expert is fetched at most
        once per forward. Splitting the token axis instead costs one launch per
        chunk -- one per token once capacity approaches ``top_k`` -- and
        refetches experts as the cache thrashes.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs.

        Returns:
            Groups of global expert ids, each no larger than ``capacity``. A
            single group when everything already fits, which is the common
            case.
        """
        # Negative ids are the kernels' "skip" marker (masked or padded
        # entries), not experts -- they must never reach the loaders. In the
        # full-DRAM path a -1 even looked harmless, because Python indexing
        # wrapped it to the last expert; the disk path turns it into a
        # negative file offset. Filter at the boundary for both.
        unique = [e for e in topk_ids.unique().tolist() if e >= 0]
        if len(unique) <= self.capacity:
            return [unique]
        return [
            unique[i : i + self.capacity] for i in range(0, len(unique), self.capacity)
        ]

    @torch.compiler.disable
    def prepare(
        self, topk_ids: torch.Tensor, unique_ids: list[int] | None = None
    ) -> ExpertWeightResult:
        """Make a set of experts resident and return the map selecting them.

        Args:
            topk_ids: Shape ``[num_tokens, top_k]``, global expert IDs. Only
                read when ``unique_ids`` is not supplied.
            unique_ids: The experts to make resident. Both planners already
                know this, and passing it avoids a device synchronization --
                which matters, because a split forward calls this once per
                piece.

        Returns:
            ExpertWeightResult holding the GPU buffers and an ``expert_map``
            exposing exactly these experts, everything else -1.

        Raises:
            RuntimeError: if more experts are requested than the cache holds.
        """
        t_start = time.perf_counter() if self._disk_store is not None else 0.0
        if unique_ids is None:
            unique_ids = [e for e in topk_ids.unique().tolist() if e >= 0]
        assert all(e >= 0 for e in unique_ids), (
            "negative expert id reached prepare(); planners filter these"
        )
        if len(unique_ids) > self.capacity:
            raise RuntimeError(
                f"CachedWeightProvider: {len(unique_ids)} unique experts "
                f"requested but --moe-expert-cache-size={self.capacity}. "
                f"Set --moe-expert-cache-size >= {len(unique_ids)}."
            )

        self._execute_plan(self._plan_group(unique_ids))

        self._prepare_calls += 1
        if self._prepare_calls % _STATS_LOG_INTERVAL == 0:
            total = self.hits + self.misses
            if total > 0:
                logger.debug(
                    "Expert cache: %d hits, %d misses (%.1f%% hit rate)"
                    ", ram %d/%d, disk reads %d",
                    self.hits,
                    self.misses,
                    100.0 * self.hits / total,
                    self.ram_hits,
                    self.ram_hits + self.ram_misses,
                    self.ram_misses,
                )

        # Expose exactly this group. Blocking on purpose: the host mirror is
        # rewritten by the next group, so an async copy could still be reading
        # it when that happens.
        t0 = time.perf_counter() if self._disk_store is not None else 0.0
        self._mapping_host.fill_(-1)
        for expert_id in unique_ids:
            self._mapping_host[expert_id] = self._lru[expert_id][0]
        self._mapping.copy_(self._mapping_host)
        if self._disk_store is not None:
            # The blocking upload synchronizes the current stream, so in a
            # multi-piece forward this also measures the host join on the
            # previous piece's kernel -- large t_mapping means compute is
            # already hiding under the reads, small means I/O-bound.
            self.t_mapping += time.perf_counter() - t0
            self.t_prepare += time.perf_counter() - t_start

        return ExpertWeightResult(
            w1=self._buf_w13,
            w2=self._buf_w2,
            expert_map=self._mapping,
            w1_scale=self._buf_w13_scale,
            w2_scale=self._buf_w2_scale,
        )


def run_with_expert_cache(
    provider: CachedWeightProvider,
    topk_ids: torch.Tensor,
    run: Callable[[ExpertWeightResult, slice, bool], torch.Tensor],
) -> torch.Tensor:
    """Evaluate a MoE forward through the expert cache.

    ``run`` receives the resident weights with the ``expert_map`` selecting
    them, the rows it should evaluate, and whether it should include work that
    belongs to the forward as a whole rather than to this call -- shared
    experts, most importantly. It sees the original ``topk_ids``; the map is
    what restricts each call to the resident experts.

    The two splits differ only in what is cut. ``"token"`` cuts rows and
    concatenates, so each token's sum stays inside one kernel call and the
    result matches the uncached path exactly. ``"expert"`` cuts the expert set
    and sums, which costs far fewer launches when capacity is small but rounds
    each group's partial sum to the model dtype.

    When everything fits -- the common case for both splits -- ``run`` is
    called exactly once and its result returned untouched.
    """
    if provider.split == "expert":
        groups = provider.plan_expert_groups(topk_ids)
        if len(groups) == 1:
            return run(provider.prepare(topk_ids, groups[0]), _ALL_ROWS, True)

        # Accumulate in fp32. It does not recover what each group already lost
        # rounding to the model dtype, but it keeps the sum from losing more.
        accumulator: torch.Tensor | None = None
        out_dtype: torch.dtype | None = None
        for i, expert_ids in enumerate(groups):
            part = run(provider.prepare(topk_ids, expert_ids), _ALL_ROWS, i == 0)
            if accumulator is None:
                accumulator, out_dtype = part.float(), part.dtype
            else:
                accumulator += part.float()
        assert accumulator is not None and out_dtype is not None
        return accumulator.to(out_dtype)

    plan = provider.plan_chunks(topk_ids)
    if len(plan) == 1:
        rows, unique_ids = plan[0]
        return run(provider.prepare(topk_ids, unique_ids), rows, True)
    return torch.cat(
        [
            run(provider.prepare(topk_ids[rows], unique_ids), rows, True)
            for rows, unique_ids in plan
        ],
        dim=0,
    )
