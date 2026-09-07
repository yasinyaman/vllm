# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Marginal-value working-set allocation (MV-WSA) for the expert/KV split.

One GPU byte can be a KV block or an expert slot, never both. Today that
split is frozen at startup: ``--moe-expert-cache-size`` takes its slots and
whatever survives profiling becomes the KV pool. A single split is the best
*compromise* over a trace, and an agentic session is not one regime -- long
context turns are KV bound, tool-call bursts are expert bound.

This module owns the decision only. It is deliberately free of torch, of
vLLM imports, and of any engine state, so the allocation law can be tested
on a laptop; ``vllm.v1.core.kv_pool_resize`` and
``CachedWeightProvider.resize`` own the physical moves.

Two allocation laws ship, because they disagree about which side has the
interesting marginal-value curve.

``expert-union`` (default). The expert side's value function is a CLIFF, not
a curve. A batch whose expert union exceeds the GPU slot count chunks and
refetches *every step* -- ``routed_experts.py`` warns about exactly this, and
its comment carries the measured number: on Qwen3-30B serving, 32 -> 64 slots
alone was worth 2.4x. Above the union, another slot buys nothing: the batch
already fits. So the correct rule is to give the expert cache exactly the
working set it is observed to need and hand every remaining byte to KV, whose
value (longer contexts, more concurrency, more surviving prefix cache) really
is continuous. The online part is that the union depends on the *live*
concurrency, not on ``max_num_seqs``: at concurrency 1 it is ``top_k``, at 16
it is ``16 * top_k``, and the gap between those is most of the pool.

``kv-peak`` (paper parity). WiSP's rule, kept for reproduction: size KV to
the observed peak plus headroom and give the remainder to the experts. Ported
faithfully, including its bias -- it treats KV as the side that saturates.
Note that its demand signal needs care under prefix caching; see
``kv_demand_blocks`` on ``Observation``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

__all__ = [
    "SplitGeometry",
    "Observation",
    "Decision",
    "MVWSAPolicy",
    "STRATEGIES",
]

STRATEGIES = ("expert-union", "kv-peak")


@dataclass(frozen=True)
class SplitGeometry:
    """Byte prices, the fixed pool, and hard bounds. Set once at startup.

    ``kv_block_bytes`` is the cost of one *logical* KV block, which reserves a
    slot in every layer's KV tensor -- so it is the sum over layers, not one
    layer's page. ``expert_slot_bytes`` is likewise summed over every MoE layer
    that owns a provider, since capacity moves in lockstep across layers.

    ``total_budget_bytes`` is the iso-VRAM pool, and it is a *constant*, not
    something re-derived from the live split each epoch. Deriving it would
    ratchet: both sides are integer counts, so every applied move rounds down
    and loses up to ``kv_block_bytes - 1`` bytes, and re-deriving the budget
    from the rounded-down split banks that loss permanently. Over a long
    session the pool would shrink move by move with nothing to show for it.
    """

    kv_block_bytes: int
    expert_slot_bytes: int
    total_budget_bytes: int
    #: Admission floor: the engine must be able to seat one request of
    #: ``max_model_len``, or it can never make progress. ceil(max_model_len /
    #: block_size), plus whatever the scheduler reserves.
    kv_floor_blocks: int
    cap_min: int
    cap_max: int

    def __post_init__(self) -> None:
        if self.kv_block_bytes <= 0:
            raise ValueError("kv_block_bytes must be positive")
        if self.expert_slot_bytes <= 0:
            raise ValueError("expert_slot_bytes must be positive")
        if self.total_budget_bytes <= 0:
            raise ValueError("total_budget_bytes must be positive")
        if self.kv_floor_blocks < 2:
            raise ValueError("kv_floor_blocks must be >= 2")
        if not 1 <= self.cap_min <= self.cap_max:
            raise ValueError("require 1 <= cap_min <= cap_max")

    def cost(self, kv_blocks: int, cap: int) -> int:
        """Bytes a given split consumes."""
        return kv_blocks * self.kv_block_bytes + cap * self.expert_slot_bytes

    @property
    def feasible(self) -> bool:
        """Can the floor and cap_min be seated inside the budget at all?"""
        return self.cost(self.kv_floor_blocks, self.cap_min) <= self.total_budget_bytes


@dataclass(frozen=True)
class Observation:
    """What the last epoch (one busy period between drained barriers) showed.

    ``kv_demand_blocks`` must be *true* demand, which is not
    ``num_gpu_blocks - get_num_free_blocks()``. With prefix caching on -- the
    default -- a cached-but-unreferenced block sits in the free queue
    (``BlockPool.free_blocks`` appends hashed ref_cnt==0 blocks back), so the
    naive difference reads the entire prefix cache as slack. A controller fed
    that number shrinks KV to the live set, discards the cache the next turn
    was about to hit, re-prefills, measures low demand again, and ratchets.
    The caller is responsible for adding back the cached blocks it wants to
    keep.

    ``expert_union_peak`` is the largest number of distinct experts a single
    MoE forward needed this epoch, maxed over layers. That is exactly the
    quantity capacity has to cover, which is what makes it a legitimate
    control input: it moves when the actuator moves.

    ``kv_pressure`` must mean "a request was actually blocked for want of KV
    blocks" -- a preemption, or an allocation that was refused. It must NOT
    mean "the waiting queue was non-empty", which is true under any concurrent
    load for reasons that have nothing to do with KV (running-request cap,
    token budget, remote KV, LoRA slots) and would ratchet capacity to the
    floor on every busy server.
    """

    kv_blocks_now: int
    cap_now: int
    kv_demand_blocks: int
    expert_union_peak: int
    kv_pressure: bool = False
    #: Epoch length in engine steps. Very short epochs are noise; the policy
    #: refuses to act on them.
    steps: int = 0


@dataclass(frozen=True)
class Decision:
    kv_from: int
    kv_to: int
    cap_from: int
    cap_to: int
    applied: bool
    reason: str
    #: The split the law wanted, even when hysteresis refused it. Reporting
    #: only; ``kv_to``/``cap_to`` are what the caller applies, and on a hold
    #: they equal the live split exactly.
    kv_target: int = 0
    cap_target: int = 0
    #: True when experts grow, which fixes the order of the physical moves:
    #: shrink the other side first, always, so the transient peak never
    #: exceeds the budget.
    experts_grow: bool = False

    @property
    def is_noop(self) -> bool:
        return self.kv_to == self.kv_from and self.cap_to == self.cap_from


@dataclass
class MVWSAPolicy:
    """Stateful decision law. One instance per engine.

    The hysteresis exists because both physical moves are expensive and, in
    the re-capture variant, visible to the user as a stall. ``deadzone_cap``
    ignores small target changes; ``confirm_epochs`` requires the same target
    to survive consecutive drains before acting, which kills flapping between
    two adjacent splits.

    Genuine KV starvation skips both, but only when the move it wants would
    actually *grow* KV. Under ``expert-union`` a pressured epoch can perfectly
    well call for a smaller KV pool -- the union went up, so the experts need
    the bytes more -- and letting "KV is starved" fast-path a KV shrink would
    invert the escape hatch it is named after.
    """

    geometry: SplitGeometry
    strategy: str = "expert-union"
    #: Fraction of slack left above measured KV demand, so the next epoch's
    #: growth does not immediately re-trigger a move.
    headroom: float = 0.15
    deadzone_cap: int = 2
    confirm_epochs: int = 2
    min_epoch_steps: int = 8

    _pending_cap: int | None = field(default=None, init=False, repr=False)
    _pending_count: int = field(default=0, init=False, repr=False)
    log: list[Decision] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(
                f"unknown MV-WSA strategy {self.strategy!r}; "
                f"expected one of {STRATEGIES}"
            )
        if not 0.0 <= self.headroom < 4.0:
            raise ValueError("headroom must be in [0, 4)")
        if self.deadzone_cap < 1:
            raise ValueError("deadzone_cap must be >= 1")
        if self.confirm_epochs < 1:
            raise ValueError("confirm_epochs must be >= 1")

    # ------------------------------------------------------------------
    # target selection
    # ------------------------------------------------------------------

    def _target(self, obs: Observation) -> tuple[int, int]:
        """Return ``(kv_blocks, cap)`` for this epoch, ignoring hysteresis.

        Both branches end at the same clamp so that the returned pair always
        fits the budget: we pick one side, derive the other from the leftover
        bytes, clamp it, then re-derive the first from the clamped value.
        Deriving in only one direction lets a clamp silently overcommit, which
        is an OOM at the next fill.
        """
        g = self.geometry
        budget = g.total_budget_bytes

        if self.strategy == "expert-union":
            # Experts get exactly their observed working set; KV takes the
            # rest. Below the union the batch chunks and refetches; above it
            # the extra slots are dead weight.
            cap_want = _clamp(obs.expert_union_peak, g.cap_min, g.cap_max)
            kv_want = (budget - cap_want * g.expert_slot_bytes) // g.kv_block_bytes
            # ... but never below what KV was observed to need. The first A/B
            # (OLMoE, agentic trace, 2026-09-07) shrank KV 1024 -> 352 blocks
            # to seat 7 more experts, and the long-context turns then queued:
            # per-request latency fell 35-45% while throughput fell 10%. KV's
            # value is continuous only above its demand; below it is a cliff
            # too. Under pressure the live pool is the demand.
            demand = obs.kv_blocks_now if obs.kv_pressure else obs.kv_demand_blocks
            kv_need = max(g.kv_floor_blocks, ceil(demand * (1.0 + self.headroom)))
            if kv_want < kv_need:
                # The floor outranks the expert working set: an engine that
                # cannot seat one max-length request makes no progress at all,
                # and one that cannot seat its live contexts preempts.
                cap_want = (budget - kv_need * g.kv_block_bytes) // g.expert_slot_bytes
        else:  # kv-peak, WiSP parity
            demand = obs.kv_blocks_now if obs.kv_pressure else obs.kv_demand_blocks
            kv_want = max(g.kv_floor_blocks, ceil(demand * (1.0 + self.headroom)))
            # Never starve the experts below cap_min to feed KV.
            kv_ceiling = (budget - g.cap_min * g.expert_slot_bytes) // g.kv_block_bytes
            kv_want = min(kv_want, kv_ceiling)
            cap_want = (budget - kv_want * g.kv_block_bytes) // g.expert_slot_bytes

        cap = _clamp(int(cap_want), g.cap_min, g.cap_max)
        kv = (budget - cap * g.expert_slot_bytes) // g.kv_block_bytes
        kv = max(2, int(kv))
        return kv, cap

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def decide(self, obs: Observation) -> Decision:
        """Fold one epoch's observation into a decision.

        Pure apart from the confirmation counter: the caller applies (or
        skips) the result and is free to ignore it. Every non-applied decision
        is an exact no-op -- ``kv_to``/``cap_to`` equal the live split -- so a
        caller that blindly applies whatever comes back cannot be walked into
        an overcommit by a hold. The split the law *wanted* is reported
        separately in ``kv_target``/``cap_target``.
        """
        g = self.geometry

        def _hold(reason: str, kv_t: int | None = None, cap_t: int | None = None):
            d = Decision(
                kv_from=obs.kv_blocks_now,
                kv_to=obs.kv_blocks_now,
                cap_from=obs.cap_now,
                cap_to=obs.cap_now,
                applied=False,
                reason=reason,
                kv_target=obs.kv_blocks_now if kv_t is None else kv_t,
                cap_target=obs.cap_now if cap_t is None else cap_t,
            )
            self.log.append(d)
            return d

        if not g.feasible:
            # cap_min plus the admission floor do not fit. Nothing this
            # controller does can help; startup sizing was wrong.
            return _hold("infeasible: cap_min + kv_floor exceeds budget")

        kv_to, cap_to = self._target(obs)

        # A starved KV pool skips the waiting, but only when the law actually
        # wants to hand KV more blocks. Pressure is not a licence to shrink it.
        urgent = obs.kv_pressure and kv_to > obs.kv_blocks_now

        if not urgent:
            if obs.steps < self.min_epoch_steps:
                # A one-step epoch says nothing about a working set. Hold
                # without touching the counter: clearing it here lets an
                # alternating short/long workload never reach confirmation.
                return _hold(f"epoch too short ({obs.steps} steps)", kv_to, cap_to)

            if abs(cap_to - obs.cap_now) < self.deadzone_cap:
                self._pending_cap = None
                self._pending_count = 0
                return _hold("within deadzone", kv_to, cap_to)

            close = self._pending_cap is not None and (
                abs(cap_to - self._pending_cap) < self.deadzone_cap
            )
            if close:
                self._pending_count += 1
            else:
                self._pending_cap = cap_to
                self._pending_count = 1
            if self._pending_count < self.confirm_epochs:
                return _hold(
                    f"unconfirmed ({self._pending_count}/{self.confirm_epochs})",
                    kv_to,
                    cap_to,
                )

        self._pending_cap = None
        self._pending_count = 0

        spend = g.cost(kv_to, cap_to)
        if spend > g.total_budget_bytes:
            # Defensive: the clamp order above should make this unreachable.
            return _hold(f"overcommit guard: {spend} > {g.total_budget_bytes}")

        d = Decision(
            kv_from=obs.kv_blocks_now,
            kv_to=kv_to,
            cap_from=obs.cap_now,
            cap_to=cap_to,
            applied=True,
            reason="ok" + (" (kv pressure)" if urgent else ""),
            kv_target=kv_to,
            cap_target=cap_to,
            experts_grow=cap_to > obs.cap_now,
        )
        self.log.append(d)
        return d


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))
