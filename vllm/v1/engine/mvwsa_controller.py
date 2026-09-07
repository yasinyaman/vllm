# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine half of MV-WSA: observe every step, re-split at drained barriers.

One GPU byte is a KV block or an expert slot, never both, and today the split
is frozen at startup. This controller moves it while serving. Each engine
step costs one subtraction (the live KV block count); at a busy->idle
transition it asks the workers for the widest expert union of the epoch,
folds that into ``MVWSAPolicy``, and applies the decision through two
actuators in the memory-safe order: ``resize_block_pool`` on the scheduler
and ``Worker.mvwsa_apply`` on the device.

Where it fires. ``EngineCoreProc`` calls ``maybe_rebalance()`` as its busy
loop turns idle, so a move -- which rebuilds the KV tensors -- lands in idle
time rather than on a request. The in-process client never enters that
loop; a harness driving ``LLM()`` with multiprocessing off calls
``maybe_rebalance(force=True)`` between generate calls itself.

What it refuses. A pool with pinned blocks (sink attention) cannot have its
prefix cache reset after a rebuild, and one layer whose kernel cannot follow
a moving buffer pins every other, so either disables the controller for good
at the first barrier, reason logged once. A move whose worker half fails is
rolled back on the device; the scheduler's pool is then reconciled to what
the worker reports in force, so the two never disagree.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.kv_pool_resize import (
    cached_block_ids,
    is_drained,
    pinned_block_ids,
    resize_block_pool,
)
from vllm.v1.core.mvwsa_policy import (
    Decision,
    MVWSAPolicy,
    Observation,
    SplitGeometry,
)

if TYPE_CHECKING:
    from vllm.v1.engine.core import EngineCore

logger = init_logger(__name__)


class MVWSAController:
    def __init__(
        self,
        engine: EngineCore,
        strategy: str,
        log_path: str | None = None,
        *,
        headroom_bytes: int = 0,
    ) -> None:
        self.engine = engine
        self.strategy = strategy
        self.policy: MVWSAPolicy | None = None
        self.cap_now: int | None = None
        self.disabled: str | None = None
        self.moves = 0
        self._steps = 0
        self._kv_live_peak = 0
        self._preemptions_seen = 0
        self._epoch_open = False
        #: Why the last maybe_rebalance() did not run a cycle; diagnostics only.
        self.last_hold: str | None = None
        # Bytes held back from the iso-VRAM budget. Zero by default: an expert
        # grow reallocates one layer's buffer at a time, so its transient is
        # one layer's w13 rows, which the memory above the pools absorbs.
        self._headroom_bytes = headroom_bytes
        self._log = open(log_path, "a") if log_path else None  # noqa: SIM115

    # ------------------------------------------------------------------
    # per-step, O(1)
    # ------------------------------------------------------------------

    def observe_step(self) -> None:
        if self.disabled:
            return
        self._steps += 1
        self._epoch_open = True
        pool = self._pool
        # Referenced blocks are never in the free queue and cached-but-
        # unreferenced ones always are, so this is exactly the live set (plus
        # pinned sinks, which never move). The prefix cache is added back at
        # the barrier; see Observation.kv_demand_blocks for why it must be.
        live = pool.num_gpu_blocks - 1 - pool.get_num_free_blocks()
        if live > self._kv_live_peak:
            self._kv_live_peak = live

    @property
    def _pool(self):
        return self.engine.scheduler.kv_cache_manager.block_pool

    def idle(self) -> bool:
        engine = self.engine
        return not (
            getattr(engine, "engines_running", False)
            or engine.scheduler.has_requests()
            or bool(getattr(engine, "batch_queue", None))
        )

    # ------------------------------------------------------------------
    # barrier
    # ------------------------------------------------------------------

    def maybe_rebalance(self, force: bool = False) -> Decision | None:
        """Run one observe -> decide -> apply cycle if the engine is drained.

        Returns the decision, or None when no cycle ran. Cheap to call every
        loop iteration: after the first idle iteration ``_epoch_open`` is
        false until the next step.
        """
        if self.disabled:
            return self._hold(f"disabled: {self.disabled}")
        if not (self._epoch_open or force):
            return self._hold("no step since the last barrier")
        scheduler = self.engine.scheduler
        # The forced path (in-process harnesses) cannot wait for the busy
        # loop's idle transition: finished-request ids stay queued for the
        # next step's outputs and keep has_requests() true. They hold no
        # blocks, so "no live request" is the right gate there; the drained
        # check below still guards the pool itself.
        busy = scheduler.has_unfinished_requests() if force else not self.idle()
        if busy:
            return self._hold("engine has work")
        if scheduler.deferred_frees:
            return self._hold("deferred frees pending")
        pool = self._pool
        if not is_drained(pool):
            return self._hold("a request still holds KV blocks")
        self._epoch_open = False
        self.last_hold = None
        if self.policy is None and not self._init_policy():
            return None
        assert self.policy is not None and self.cap_now is not None

        union = max(self.engine.collective_rpc("mvwsa_take_union_peak"), default=0)
        preempted = getattr(scheduler, "num_preemptions", 0)
        obs = Observation(
            kv_blocks_now=pool.num_gpu_blocks,
            cap_now=self.cap_now,
            kv_demand_blocks=self._kv_live_peak + len(cached_block_ids(pool)),
            expert_union_peak=union,
            kv_pressure=preempted > self._preemptions_seen,
            steps=self._steps,
        )
        self._preemptions_seen = preempted
        self._steps = 0
        self._kv_live_peak = 0

        decision = self.policy.decide(obs)
        report = None
        if decision.applied and not decision.is_noop:
            report = self._apply(decision)
        self._write(obs, decision, report)
        return decision

    def _hold(self, reason: str) -> None:
        self.last_hold = reason
        return None

    def _init_policy(self) -> bool:
        engine = self.engine
        pool = self._pool
        g = engine.collective_rpc("mvwsa_geometry")[0]
        if not g["resizable"]:
            return self._disable(
                "a layer's expert cache cannot resize (no ceiling, zero-copy, "
                "or a backend that captures its buffers)"
            )
        if pinned_block_ids(pool):
            return self._disable(
                "the KV pool has permanently pinned blocks (attention sinks); "
                "a rebuilt cache could not reset its prefix entries"
            )
        vc = engine.vllm_config
        groups = len(engine.scheduler.kv_cache_config.kv_cache_groups) or 1
        block_size = vc.cache_config.block_size
        max_len = vc.model_config.max_model_len
        kv_floor = (
            groups * (-(-max_len // block_size))
            + 1
            + max(vc.scheduler_config.max_num_seqs, 1)
        )
        kv0, cap0 = pool.num_gpu_blocks, g["capacity"]
        budget = (
            kv0 * g["kv_block_bytes"]
            + cap0 * g["expert_slot_bytes"]
            - self._headroom_bytes
        )
        try:
            geometry = SplitGeometry(
                kv_block_bytes=g["kv_block_bytes"],
                expert_slot_bytes=g["expert_slot_bytes"],
                total_budget_bytes=budget,
                kv_floor_blocks=max(2, kv_floor),
                cap_min=g["cap_min"],
                cap_max=g["cap_max"],
            )
        except ValueError as exc:
            return self._disable(f"geometry rejected: {exc}")
        self.policy = MVWSAPolicy(geometry=geometry, strategy=self.strategy)
        self.cap_now = cap0
        if not geometry.feasible:
            logger.warning(
                "MV-WSA: cap_min=%d plus the admission floor of %d blocks do not "
                "fit the %d-byte budget; the controller will hold every epoch.",
                geometry.cap_min,
                geometry.kv_floor_blocks,
                budget,
            )
        logger.info(
            "MV-WSA %s: kv=%d blocks x %d B, cap=%d slots x %d B (floor %d, "
            "range %d..%d), budget %d B",
            self.strategy,
            kv0,
            g["kv_block_bytes"],
            cap0,
            g["expert_slot_bytes"],
            geometry.kv_floor_blocks,
            g["cap_min"],
            g["cap_max"],
            budget,
        )
        return True

    def _disable(self, reason: str) -> bool:
        self.disabled = reason
        logger.warning("MV-WSA disabled: %s", reason)
        return False

    # ------------------------------------------------------------------
    # apply
    # ------------------------------------------------------------------

    def _apply(self, d: Decision) -> dict[str, Any]:
        scheduler = self.engine.scheduler
        pool = self._pool
        manager = scheduler.kv_cache_manager
        if d.experts_grow:
            # KV gives bytes up: the scheduler stops handing out the ids the
            # worker is about to drop, then the device moves.
            if d.kv_to != d.kv_from:
                resize_block_pool(pool, d.kv_to, manager)
            report = self.engine.collective_rpc(
                "mvwsa_apply", args=(d.kv_to, d.cap_to, True)
            )[0]
        else:
            # Experts give bytes up: the device grows KV first, then the
            # scheduler may address the new ids.
            report = self.engine.collective_rpc(
                "mvwsa_apply", args=(d.kv_to, d.cap_to, False)
            )[0]
            if report["failed"] is None and d.kv_to != d.kv_from:
                resize_block_pool(pool, d.kv_to, manager)

        self.cap_now = report["cap_in_force"]
        kv_in_force = report["kv_in_force"]
        if pool.num_gpu_blocks != kv_in_force:
            # A failed half left the worker at a different size; follow it.
            resize_block_pool(pool, kv_in_force, manager)
        self._sync_num_blocks(kv_in_force)

        kv_report = report.get("kv")
        rebuilt = kv_report is not None and not kv_report.get(
            "content_preserved", False
        )
        # The tensors came back zeroed; every cached hash now points at zeros.
        # Reset before the next request can hit one.
        if rebuilt and not scheduler.reset_prefix_cache():
            raise RuntimeError(
                "MV-WSA: prefix cache could not be reset after a KV rebuild"
            )
        if report["failed"]:
            logger.warning(
                "MV-WSA move %s failed on the worker (%s); in force: kv=%d cap=%d",
                (d.kv_from, d.kv_to, d.cap_from, d.cap_to),
                report["failed"],
                kv_in_force,
                self.cap_now,
            )
        else:
            self.moves += 1
            logger.info(
                "MV-WSA move %d: kv %d -> %d blocks, cap %d -> %d slots (%s)",
                self.moves,
                d.kv_from,
                d.kv_to,
                d.cap_from,
                d.cap_to,
                d.reason,
            )
        return report

    def _sync_num_blocks(self, num_blocks: int) -> None:
        # The three copies the rest of the engine reads: the engine's
        # cache_config (Prometheus, flex attention), the scheduler's own
        # kv_cache_config, and the pool itself (already moved).
        self.engine.vllm_config.cache_config.num_gpu_blocks = num_blocks
        self.engine.scheduler.kv_cache_config.num_blocks = num_blocks

    def _write(
        self, obs: Observation, d: Decision, report: dict[str, Any] | None
    ) -> None:
        if self._log is None:
            return
        self._log.write(
            json.dumps(
                {
                    "t": time.time(),
                    "moves": self.moves,
                    "obs": asdict(obs),
                    "decision": asdict(d),
                    "report": report,
                }
            )
            + "\n"
        )
        self._log.flush()
