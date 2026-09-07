# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the live GPU slot-tier resize (the expert half of MV-WSA).

Deliberately device-agnostic: the resize is bookkeeping plus one buffer
realloc, and none of it needs CUDA. That keeps it runnable on a laptop, which
is where the invariants get broken. The fill/kernel paths that do need CUDA
are covered by test_expert_lru_cache.py.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    expert_cache_bounds,
)

pytestmark = pytest.mark.cpu_test

E = 8  # experts
H = 4
INTER = 2


@pytest.fixture(autouse=True)
def _allocate_on_cpu(monkeypatch):
    """Keep the provider's buffers on CPU regardless of the host's accelerator.

    The provider picks its device from ``torch.accelerator`` when the weights
    arrive on CPU, and pins its host mirror. Neither is meaningful for a
    bookkeeping test, and on a machine whose accelerator is not CUDA the pin
    lands on the wrong device. Forcing both keeps this file runnable off a
    GPU box, which is where the slot invariants are cheapest to check.
    """
    monkeypatch.setattr(
        torch.accelerator, "current_accelerator", lambda *a, **k: torch.device("cpu")
    )
    monkeypatch.setattr(
        torch.Tensor, "pin_memory", lambda self, *a, **k: self, raising=False
    )


def make_provider(
    capacity: int,
    max_capacity: int | None = None,
    with_scales: bool = False,
    split: str = "expert",
    min_capacity: int = 1,
) -> CachedWeightProvider:
    w13 = torch.randn(E, 2 * INTER, H)
    w2 = torch.randn(E, H, INTER)
    scales = {}
    if with_scales:
        scales = dict(
            w13_scale=torch.randn(E, 2 * INTER, 1),
            w2_scale=torch.randn(E, H, 1),
        )
    return CachedWeightProvider(
        capacity=capacity,
        w13_weight=w13,
        w2_weight=w2,
        split=split,
        max_capacity=max_capacity,
        min_capacity=min_capacity,
        **scales,
    )


def seat(provider: CachedWeightProvider, expert_id: int, slot: int) -> None:
    """Mark ``expert_id`` resident in ``slot``, as _plan_group would."""
    assert slot in provider._free_slots
    provider._free_slots.remove(slot)
    provider._clock += 1
    provider._lru[expert_id] = [slot, 1, provider._clock]


def resident_map(provider: CachedWeightProvider) -> dict[int, int]:
    return {eid: entry[0] for eid, entry in provider._lru.items()}


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_max_capacity_defaults_to_capacity():
    p = make_provider(4)
    assert p.max_capacity == 4
    assert p.capacity == 4


def test_resize_is_opt_in():
    """A provider built without max_capacity must refuse to move.

    Not every backend tolerates it: XpuFusedMoe captures w13/w2 at its first
    apply() and never re-reads them, so opting in is RoutedExperts' call.
    """
    p = make_provider(4)
    with pytest.raises(RuntimeError, match="not built resizable"):
        p.resize(2)


def test_max_capacity_zero_is_rejected_not_ignored():
    """0 is a stated ceiling below capacity, not an omitted argument."""
    with pytest.raises(ValueError, match="must be >="):
        make_provider(4, max_capacity=0)


def test_max_capacity_must_not_be_below_capacity():
    with pytest.raises(ValueError, match="must be >="):
        make_provider(8, max_capacity=4)


def test_scale_buffers_are_sized_at_max_capacity():
    """They can never be reallocated, so they are built for the ceiling."""
    p = make_provider(2, max_capacity=6, with_scales=True)
    assert p.buf_w13.shape[0] == 2
    assert p.buf_w13_scale.shape[0] == 6
    assert p.buf_w2_scale.shape[0] == 6


def test_slot_bytes_counts_weights_only():
    p = make_provider(4, with_scales=True)
    expected = p.buf_w13[0].nbytes + p.buf_w2[0].nbytes
    assert p.slot_bytes == expected


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_refuses_growth_past_max_capacity():
    p = make_provider(4, max_capacity=6)
    with pytest.raises(ValueError, match="exceeds max_capacity"):
        p.resize(7)


def test_refuses_capacity_below_the_floor():
    p = make_provider(4, max_capacity=8)
    with pytest.raises(ValueError, match="below min_capacity"):
        p.resize(0)


def test_token_split_floor_is_enforced():
    """Under the token split one token's whole expert set must fit."""
    p = make_provider(8, max_capacity=8, split="token", min_capacity=4)
    assert p.resize(4) == 4
    with pytest.raises(ValueError, match="below min_capacity"):
        p.resize(3)


def test_grow_releases_each_old_buffer_before_allocating_the_next():
    """Peak during a grow must not hold both full buffer pairs at once."""
    p = make_provider(2, max_capacity=64)
    seen = []
    real_empty = torch.empty

    def spy(*args, **kwargs):
        seen.append((p._buf_w13.shape[0], p._buf_w2.shape[0]))
        return real_empty(*args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(torch, "empty", spy):
        p.resize(64)
    # At the second allocation w13 has already been swapped to the new size,
    # i.e. the old w13 was released before w2 was allocated.
    assert seen[0] == (2, 2)
    assert seen[1] == (64, 2)


def test_same_capacity_is_a_noop():
    p = make_provider(4, max_capacity=8)
    seat(p, 3, 0)
    before = p.buf_w13
    assert p.resize(4) == 4
    assert p.buf_w13 is before
    assert resident_map(p) == {3: 0}


# ---------------------------------------------------------------------------
# grow
# ---------------------------------------------------------------------------


def test_grow_extends_slots_and_keeps_residency():
    p = make_provider(2, max_capacity=6)
    seat(p, 5, 0)
    seat(p, 6, 1)
    assert p.resize(5) == 5
    assert p.capacity == 5
    assert p.buf_w13.shape[0] == 5
    assert p.buf_w2.shape[0] == 5
    # Nothing re-pages merely because the cache grew.
    assert resident_map(p) == {5: 0, 6: 1}
    assert sorted(p._free_slots) == [2, 3, 4]


def test_grow_copies_live_slot_bytes_forward():
    p = make_provider(2, max_capacity=6)
    seat(p, 5, 0)
    p.buf_w13[0].fill_(3.5)
    p.buf_w2[0].fill_(-1.25)
    p.resize(5)
    assert torch.all(p.buf_w13[0] == 3.5)
    assert torch.all(p.buf_w2[0] == -1.25)


def test_grow_leaves_scale_buffer_identity_intact():
    """The kernel captured these tensors; repointing would be silently wrong."""
    p = make_provider(2, max_capacity=6, with_scales=True)
    w13_scale, w2_scale = p.buf_w13_scale, p.buf_w2_scale
    p.resize(6)
    assert p.buf_w13_scale is w13_scale
    assert p.buf_w2_scale is w2_scale


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------


def test_shrink_drops_only_the_experts_that_lose_their_slot():
    p = make_provider(6, max_capacity=6)
    for eid, slot in enumerate(range(6)):
        seat(p, eid, slot)
    assert p.resize(3) == 3
    assert resident_map(p) == {0: 0, 1: 1, 2: 2}
    assert p._free_slots == []


def test_shrink_keeps_surviving_slot_bytes():
    p = make_provider(4, max_capacity=4)
    seat(p, 1, 0)
    p.buf_w13[0].fill_(7.0)
    p.resize(2)
    assert torch.all(p.buf_w13[0] == 7.0)
    assert resident_map(p) == {1: 0}


def test_shrink_returns_freed_slots_correctly():
    p = make_provider(6, max_capacity=6)
    seat(p, 0, 0)
    seat(p, 1, 4)  # lives in the doomed tail
    assert sorted(p._free_slots) == [1, 2, 3, 5]
    p.resize(3)
    assert resident_map(p) == {0: 0}
    assert sorted(p._free_slots) == [1, 2]


def test_shrink_routes_evictions_through_the_policy():
    p = make_provider(4, max_capacity=4)
    seen = []
    p._gpu_policy.on_evict = lambda eid: seen.append(eid)  # type: ignore[method-assign]
    seat(p, 2, 0)
    seat(p, 3, 3)
    p.resize(2)
    assert seen == [3], "only the expert that lost its slot should be evicted"


def test_shrink_leaves_scale_buffer_identity_intact():
    p = make_provider(6, max_capacity=6, with_scales=True)
    w13_scale, w2_scale = p.buf_w13_scale, p.buf_w2_scale
    p.resize(2)
    assert p.buf_w13_scale is w13_scale
    assert p.buf_w2_scale is w2_scale
    # Still addressable for every live slot.
    assert p.buf_w13_scale.shape[0] >= p.capacity


# ---------------------------------------------------------------------------
# invariants across sequences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sequence", [(1, 8, 1), (8, 1, 8), (4, 5, 3, 8, 2), (2, 2, 8, 8, 1)]
)
def test_invariants_hold_across_resize_sequences(sequence):
    p = make_provider(4, max_capacity=8)
    seat(p, 0, 0)
    seat(p, 1, 1)
    for target in sequence:
        p.resize(target)
        assert p.capacity == target
        assert p.buf_w13.shape[0] == target
        assert p.buf_w2.shape[0] == target
        p._assert_slot_invariants()


def test_slots_never_double_book_after_churn():
    p = make_provider(8, max_capacity=8)
    for eid in range(8):
        seat(p, eid, eid)
    p.resize(3)
    p.resize(8)
    # The three survivors kept their slots; the rest came back as free.
    assert resident_map(p) == {0: 0, 1: 1, 2: 2}
    assert sorted(p._free_slots) == [3, 4, 5, 6, 7]
    p._assert_slot_invariants()


def test_mapping_tensor_is_untouched_by_resize():
    """The device expert_map is num_experts-sized, not capacity-sized."""
    p = make_provider(4, max_capacity=8)
    mapping = p._mapping
    p.resize(8)
    assert p._mapping is mapping
    assert p._mapping.shape[0] == E


# ----------------------------------------------------------------------
# The controller's expert-side input: the per-call union peak.
# ----------------------------------------------------------------------


def test_union_peak_tracks_the_widest_call_and_resets_on_take():
    p = make_provider(4, max_capacity=8, split="expert")
    assert p.union_peak == 0
    p.plan_expert_groups(torch.tensor([[0, 1, 2]], dtype=torch.int32))
    p.plan_expert_groups(torch.tensor([[0, 1], [5, 6], [7, 7]], dtype=torch.int32))
    p.plan_expert_groups(torch.tensor([[3]], dtype=torch.int32))
    assert p.union_peak == 5  # {0,1,5,6,7}
    assert p.take_union_peak() == 5
    assert p.union_peak == 0
    assert p.take_union_peak() == 0


def test_union_peak_counts_the_token_split_too():
    p = make_provider(4, max_capacity=8, split="token", min_capacity=2)
    p.plan_chunks(torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32))
    assert p.take_union_peak() == 6


def test_union_peak_ignores_masked_ids():
    p = make_provider(4, max_capacity=8, split="expert")
    p.plan_expert_groups(torch.tensor([[0, -1, -1, 2]], dtype=torch.int32))
    assert p.take_union_peak() == 2


# ----------------------------------------------------------------------
# expert_cache_bounds: what RoutedExperts hands the provider.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,max_size,local,top_k,split,ram,ok,expect",
    [
        # no ceiling asked for -> fixed cache, floor still reported
        (16, 0, 64, 8, "token", 0, True, (16, 8, None)),
        (4, 0, 64, 8, "expert", 0, True, (4, 1, None)),
        # ceiling clamps to the experts the layer has
        (16, 128, 64, 8, "token", 0, True, (16, 8, 64)),
        # under a disk tier the RAM tier is the real ceiling
        (16, 64, 64, 8, "token", 32, True, (16, 8, 32)),
        # backend cannot follow a moving buffer -> fixed
        (16, 64, 64, 8, "token", 0, False, (16, 8, None)),
        # size above local experts is clamped, ceiling never below capacity
        (100, 100, 64, 8, "expert", 0, True, (64, 1, 64)),
        (32, 64, 64, 8, "token", 16, True, (32, 8, 32)),
    ],
)
def test_expert_cache_bounds(size, max_size, local, top_k, split, ram, ok, expect):
    assert (
        expert_cache_bounds(
            size, max_size, local, top_k, split, ram_capacity=ram, backend_can_resize=ok
        )
        == expect
    )
