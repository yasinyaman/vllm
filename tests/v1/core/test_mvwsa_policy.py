# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the MV-WSA expert/KV allocation law.

The policy is deliberately torch-free, so these run anywhere. The physical
moves it drives are covered by tests/v1/core/test_kv_pool_resize.py and the
expert-cache resize suites.
"""

import pytest

from vllm.v1.core.mvwsa_policy import (
    Decision,
    MVWSAPolicy,
    Observation,
    SplitGeometry,
)

# Qwen3-30B-A3B-ish shapes on a 24 GiB card. The odd byte counts are
# deliberate: they make integer floor-division lose a few bytes on every
# move, which is what a ratcheting budget would compound.
KV_BLOCK = (1 << 20) + 7
SLOT = (18 << 20) + 13
KV0, CAP0 = 2048, 32
BUDGET = KV0 * KV_BLOCK + CAP0 * SLOT


def geom(**kw) -> SplitGeometry:
    base = dict(
        kv_block_bytes=KV_BLOCK,
        expert_slot_bytes=SLOT,
        total_budget_bytes=BUDGET,
        kv_floor_blocks=64,
        cap_min=8,
        cap_max=128,
    )
    base.update(kw)
    return SplitGeometry(**base)


def obs(**kw) -> Observation:
    base = dict(
        kv_blocks_now=KV0,
        cap_now=CAP0,
        kv_demand_blocks=400,
        expert_union_peak=32,
        kv_pressure=False,
        steps=64,
    )
    base.update(kw)
    return Observation(**base)


def settled(policy: MVWSAPolicy, observation: Observation, rounds: int = 4) -> Decision:
    """Drive the same observation until the confirmation counter clears."""
    d = policy.decide(observation)
    for _ in range(rounds - 1):
        if d.applied:
            break
        d = policy.decide(observation)
    return d


# ---------------------------------------------------------------------------
# geometry invariants
# ---------------------------------------------------------------------------


def test_geometry_rejects_nonsense():
    with pytest.raises(ValueError):
        geom(kv_block_bytes=0)
    with pytest.raises(ValueError):
        geom(expert_slot_bytes=-1)
    with pytest.raises(ValueError):
        geom(total_budget_bytes=0)
    with pytest.raises(ValueError):
        geom(kv_floor_blocks=1)
    with pytest.raises(ValueError):
        geom(cap_min=64, cap_max=32)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        MVWSAPolicy(geometry=geom(), strategy="whatever")


def test_infeasible_budget_holds():
    g = geom(kv_floor_blocks=100_000, cap_min=100)
    assert not g.feasible
    p = MVWSAPolicy(geometry=g)
    d = p.decide(obs())
    assert not d.applied
    assert "infeasible" in d.reason
    assert d.is_noop


# ---------------------------------------------------------------------------
# the iso-VRAM invariant -- the one thing that must never break
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["expert-union", "kv-peak"])
@pytest.mark.parametrize("union", [1, 8, 32, 64, 128, 1024])
@pytest.mark.parametrize("demand", [0, 100, 1000, 5000])
def test_never_overcommits_the_budget(strategy, union, demand):
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy=strategy, confirm_epochs=1)
    d = p.decide(obs(expert_union_peak=union, kv_demand_blocks=demand))
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes, (
        f"{strategy} overcommitted: kv={d.kv_to} cap={d.cap_to}"
    )


@pytest.mark.parametrize("strategy", ["expert-union", "kv-peak"])
def test_bounds_are_respected(strategy):
    g = geom(cap_min=8, cap_max=64)
    p = MVWSAPolicy(geometry=g, strategy=strategy, confirm_epochs=1)
    for union in (0, 1, 200, 10_000):
        for demand in (0, 10_000, 10**9):
            d = p.decide(obs(expert_union_peak=union, kv_demand_blocks=demand))
            assert g.cap_min <= d.cap_to <= g.cap_max
            assert d.kv_to >= 2
            assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_budget_does_not_ratchet_down_across_many_moves():
    """The pool is a constant. Re-deriving it each epoch would bleed it away.

    Both sides are integer counts, so every applied move rounds down and
    loses up to kv_block_bytes-1. A controller that then treats the rounded
    split as the new budget banks that loss forever.
    """
    g = geom()
    p = MVWSAPolicy(
        geometry=g, strategy="expert-union", confirm_epochs=1, deadzone_cap=1
    )
    kv, cap = KV0, CAP0
    splits_for = {}
    for i in range(40):
        union = 16 if i % 2 == 0 else 96
        d = p.decide(obs(kv_blocks_now=kv, cap_now=cap, expert_union_peak=union))
        if d.applied:
            kv, cap = d.kv_to, d.cap_to
            # Utilisation must never decay: always within one KV block of full.
            slack = g.total_budget_bytes - g.cost(kv, cap)
            assert 0 <= slack < g.kv_block_bytes, f"leaked {slack} bytes by move {i}"
            splits_for.setdefault(union, set()).add((kv, cap))
    for union, splits in splits_for.items():
        assert len(splits) == 1, f"union {union} drifted across {sorted(splits)}"


# ---------------------------------------------------------------------------
# expert-union: the default law
# ---------------------------------------------------------------------------


def test_expert_union_sizes_cap_to_the_observed_working_set():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union")
    d = settled(p, obs(cap_now=32, expert_union_peak=64))
    assert d.applied
    assert d.cap_to == 64
    assert d.experts_grow


def test_expert_union_gives_slack_back_to_kv():
    """A quiet single-stream epoch needs few slots; KV should absorb the rest."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union")
    o = obs(cap_now=64, expert_union_peak=8, kv_demand_blocks=50)
    d = settled(p, o)
    assert d.applied
    assert d.cap_to == 8
    assert d.kv_to > o.kv_blocks_now
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_expert_union_never_breaks_the_admission_floor():
    """The floor outranks the expert working set: no floor, no progress.

    Pinned to the exact split so the test fails if the floor branch is
    removed -- without it cap would clamp to cap_max and KV would collapse
    to its 2-block minimum.
    """
    g = geom(kv_floor_blocks=1500, cap_max=1024)
    p = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=1)
    d = p.decide(obs(cap_now=32, kv_blocks_now=KV0, expert_union_peak=1024))
    expected_cap = (
        g.total_budget_bytes - g.kv_floor_blocks * g.kv_block_bytes
    ) // g.expert_slot_bytes
    expected_kv = (
        g.total_budget_bytes - expected_cap * g.expert_slot_bytes
    ) // g.kv_block_bytes
    assert d.cap_to == expected_cap
    assert d.kv_to == expected_kv
    assert d.kv_to >= g.kv_floor_blocks
    assert d.cap_to < 1024, "cap must yield to the floor, not clamp to cap_max"


def test_expert_union_is_not_a_ratchet():
    """Union up then down must come back down -- the signal tracks the actuator."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union")
    up = settled(p, obs(cap_now=16, expert_union_peak=96))
    assert up.applied and up.cap_to == 96
    down = settled(p, obs(cap_now=up.cap_to, expert_union_peak=16))
    assert down.applied and down.cap_to == 16


# ---------------------------------------------------------------------------
# kv-peak: paper parity
# ---------------------------------------------------------------------------


def test_kv_peak_reclaims_idle_kv_for_experts():
    """WiSP's headline move: KV pool mostly idle -> bytes become slots."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="kv-peak")
    o = obs(cap_now=32, kv_blocks_now=KV0, kv_demand_blocks=300)
    d = settled(p, o)
    assert d.applied
    assert d.kv_to < o.kv_blocks_now
    assert d.cap_to > o.cap_now
    assert d.experts_grow


def test_kv_peak_applies_headroom():
    g = geom(cap_max=4096)
    p = MVWSAPolicy(geometry=g, strategy="kv-peak", headroom=0.5, confirm_epochs=1)
    d = p.decide(obs(kv_demand_blocks=1000, kv_blocks_now=KV0, cap_now=8))
    assert d.kv_to >= 1500


def test_kv_peak_leaves_room_for_cap_min_under_unbounded_demand():
    """Demand larger than the pool must not squeeze the experts below cap_min."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="kv-peak", confirm_epochs=1)
    d = p.decide(obs(kv_demand_blocks=10**9, kv_blocks_now=KV0, cap_now=CAP0))
    assert d.cap_to == g.cap_min
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_kv_peak_pressure_treats_the_pool_as_saturated():
    g = geom()
    quiet = MVWSAPolicy(geometry=g, strategy="kv-peak", confirm_epochs=1).decide(
        obs(kv_demand_blocks=100, kv_blocks_now=KV0, cap_now=CAP0)
    )
    pressed = MVWSAPolicy(geometry=g, strategy="kv-peak", confirm_epochs=1).decide(
        obs(
            kv_demand_blocks=100,
            kv_blocks_now=KV0,
            cap_now=CAP0,
            kv_pressure=True,
        )
    )
    assert pressed.kv_to > quiet.kv_to


# ---------------------------------------------------------------------------
# hysteresis
# ---------------------------------------------------------------------------


def test_deadzone_suppresses_small_moves():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", deadzone_cap=4)
    d = p.decide(obs(cap_now=32, expert_union_peak=34))
    assert not d.applied
    assert d.reason == "within deadzone"


def test_confirmation_required_before_acting():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=3)
    o = obs(cap_now=32, expert_union_peak=96)
    first, second, third = p.decide(o), p.decide(o), p.decide(o)
    assert not first.applied and "unconfirmed" in first.reason
    assert not second.applied
    assert third.applied and third.cap_to == 96


def test_flapping_target_never_confirms():
    g = geom()
    p = MVWSAPolicy(
        geometry=g, strategy="expert-union", confirm_epochs=2, deadzone_cap=2
    )
    for union in (96, 16, 96, 16, 96, 16):
        d = p.decide(obs(cap_now=32, expert_union_peak=union))
        assert not d.applied, f"flapping target was applied at union={union}"


def test_short_epochs_do_not_deadlock_confirmation():
    """A short epoch is ignored, not a reset -- otherwise alternating
    short/long epochs would never accumulate a confirmation."""
    g = geom()
    p = MVWSAPolicy(
        geometry=g, strategy="expert-union", confirm_epochs=2, min_epoch_steps=8
    )
    steady = obs(cap_now=32, expert_union_peak=96)
    blip = obs(cap_now=32, expert_union_peak=96, steps=1)
    assert not p.decide(steady).applied
    assert not p.decide(blip).applied
    assert p.decide(steady).applied, "short epochs must not clear the counter"


def test_short_epoch_is_reported_and_is_a_noop():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", min_epoch_steps=8)
    d = p.decide(obs(cap_now=32, expert_union_peak=96, steps=1))
    assert not d.applied and "too short" in d.reason
    assert d.is_noop


# ---------------------------------------------------------------------------
# kv_pressure escape hatch
# ---------------------------------------------------------------------------


def test_kv_pressure_bypasses_confirmation_when_it_grows_kv():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="kv-peak", confirm_epochs=5)
    d = p.decide(
        obs(
            cap_now=64,
            kv_blocks_now=512,
            kv_demand_blocks=512,
            kv_pressure=True,
        )
    )
    assert d.applied, "a starved KV pool must not wait for confirmation"
    assert d.kv_to > 512


def test_kv_pressure_never_shrinks_kv_under_expert_union():
    """Pressure means KV was starved; the union rising does not override that.

    Before the first A/B this law let a pressured epoch call for a smaller
    pool because the expert union rose. Now the live pool is the demand
    under pressure, so the target keeps or grows KV, and a grow may take the
    fast path -- that is what the escape hatch is for.
    """
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=3)
    o = obs(cap_now=8, kv_blocks_now=KV0, expert_union_peak=128, kv_pressure=True)
    d = p.decide(o)
    assert d.kv_target >= o.kv_blocks_now
    assert d.kv_to >= o.kv_blocks_now
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_a_confirmed_kv_shrink_still_serves_its_confirmation():
    """Without pressure, a KV shrink for a wider union waits like any move."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=3)
    o = obs(cap_now=8, kv_blocks_now=KV0, kv_demand_blocks=64, expert_union_peak=128)
    d = p.decide(o)
    assert d.kv_target < o.kv_blocks_now, "test needs a KV-shrinking target"
    assert not d.applied and "unconfirmed" in d.reason
    assert d.is_noop


def test_kv_pressure_is_not_suppressed_by_the_deadzone():
    """Sustained starvation must eventually move KV even if cap barely changes."""
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="kv-peak", deadzone_cap=64, confirm_epochs=1)
    d = p.decide(
        obs(
            cap_now=CAP0,
            kv_blocks_now=KV0,
            kv_demand_blocks=KV0,
            kv_pressure=True,
        )
    )
    assert d.applied
    assert d.kv_to > KV0


# ---------------------------------------------------------------------------
# decision bookkeeping
# ---------------------------------------------------------------------------


def test_grow_direction_is_reported():
    g = geom()
    grow = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=1).decide(
        obs(cap_now=16, expert_union_peak=96)
    )
    assert grow.experts_grow is True
    shrink = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=1).decide(
        obs(cap_now=96, expert_union_peak=16)
    )
    assert shrink.experts_grow is False


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        (dict(expert_union_peak=33), "within deadzone"),
        (dict(expert_union_peak=96, steps=1), "too short"),
        (dict(expert_union_peak=96), "unconfirmed"),
    ],
)
def test_every_hold_is_an_exact_noop(kwargs, expect):
    """A caller that applies whatever comes back must never be walked into a
    move by a hold -- so a hold reports the live split, not its wish."""
    g = geom()
    p = MVWSAPolicy(
        geometry=g, strategy="expert-union", deadzone_cap=8, confirm_epochs=3
    )
    o = obs(cap_now=32, kv_blocks_now=KV0, **kwargs)
    d = p.decide(o)
    assert not d.applied and expect in d.reason
    assert d.is_noop
    assert (d.kv_to, d.cap_to) == (o.kv_blocks_now, o.cap_now)
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_holds_still_report_the_target_they_wanted():
    g = geom()
    p = MVWSAPolicy(geometry=g, strategy="expert-union", confirm_epochs=3)
    d = p.decide(obs(cap_now=32, expert_union_peak=96))
    assert not d.applied
    assert d.cap_target == 96
    assert d.cap_to == 32


# ----------------------------------------------------------------------
# expert-union must not starve KV below its observed demand
# ----------------------------------------------------------------------


def _confirm(policy, obs):
    """Drive one observation through the confirmation hysteresis."""
    d = None
    for _ in range(policy.confirm_epochs):
        d = policy.decide(obs)
    return d


def test_expert_union_keeps_kv_at_demand_plus_headroom():
    g = geom()
    policy = MVWSAPolicy(geometry=g, strategy="expert-union")
    # The union asks for every slot; the live contexts need most of the KV
    # pool. The experts get what is left above demand + headroom, not the
    # whole union.
    obs = Observation(
        kv_blocks_now=KV0,
        cap_now=CAP0,
        kv_demand_blocks=1600,
        expert_union_peak=128,
        steps=100,
    )
    d = _confirm(policy, obs)
    assert d.applied
    assert d.kv_to >= 1600 * 1.15 - 1
    assert d.cap_to < 128
    assert g.cost(d.kv_to, d.cap_to) <= g.total_budget_bytes


def test_expert_union_under_pressure_never_shrinks_kv():
    g = geom()
    policy = MVWSAPolicy(geometry=g, strategy="expert-union")
    obs = Observation(
        kv_blocks_now=KV0,
        cap_now=CAP0,
        kv_demand_blocks=100,  # a stale low reading
        expert_union_peak=128,
        kv_pressure=True,
        steps=100,
    )
    d = _confirm(policy, obs)
    assert d.kv_to >= KV0
    assert d.cap_to <= CAP0


def test_expert_union_still_hands_slack_to_experts():
    g = geom()
    policy = MVWSAPolicy(geometry=g, strategy="expert-union")
    obs = Observation(
        kv_blocks_now=KV0,
        cap_now=CAP0,
        kv_demand_blocks=64,
        expert_union_peak=64,
        steps=100,
    )
    d = _confirm(policy, obs)
    assert d.applied and d.cap_to == 64 and d.kv_to < KV0
