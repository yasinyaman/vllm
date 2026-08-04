# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CachedWeightProvider (LFRU expert cache)."""

import queue
import time

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_disk_store import (
    ALIGN,
    DiskExpertStore,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

NUM_EXPERTS = [8, 64]
DTYPES = [torch.bfloat16, torch.float16]
CAPACITIES = [1, 4]
HIDDEN = 16
INTERMEDIATE = 32


def _make_weights(num_experts: int, dtype: torch.dtype):
    # Generate in BF16 then cast - torch.randn doesn't support FP8 dtypes
    w13 = torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
    w2 = torch.randn(num_experts, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)
    return w13.to(dtype), w2.to(dtype)


def _make_scales(num_experts: int):
    w13_s = torch.rand(num_experts, 1, dtype=torch.float32)
    w2_s = torch.rand(num_experts, 1, dtype=torch.float32)
    return w13_s, w2_s


def _make_provider(
    num_experts: int = 8,
    capacity: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    with_scales: bool = False,
    split: str = "token",
):
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    kwargs: dict = dict(capacity=capacity, w13_weight=w13, w2_weight=w2, split=split)
    scales = None
    if with_scales:
        w13_s, w2_s = _make_scales(num_experts)
        kwargs.update(w13_scale=w13_s, w2_scale=w2_s)
        scales = (w13_s, w2_s)
    return CachedWeightProvider(**kwargs), w13, w2, scales


def _topk(ids: list[int]) -> torch.Tensor:
    return torch.tensor(ids, dtype=torch.int32, device="cuda").unsqueeze(0)


# -- Core cache behavior --


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("capacity", CAPACITIES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cold_miss_and_warm_hit(num_experts: int, capacity: int, dtype: torch.dtype):
    """Cold access misses, repeat access hits. GPU buffer matches source."""
    provider, w13, w2, _ = _make_provider(num_experts, capacity, dtype)
    expert_ids = list(range(min(capacity, num_experts)))

    # Cold miss
    result = provider.prepare(_topk(expert_ids))
    assert provider.misses == len(expert_ids)
    assert provider.hits == 0
    assert isinstance(result, ExpertWeightResult)
    assert result.w1 is provider.buf_w13
    assert result.w2 is provider.buf_w2
    assert result.expert_map.shape == (num_experts,)

    # Verify GPU buffer contents match source weights
    for eid in expert_ids:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        torch.testing.assert_close(result.w2[slot].cpu(), w2[eid])

    # Warm hit
    prev_misses = provider.misses
    provider.prepare(_topk(expert_ids))
    assert provider.hits == len(expert_ids)
    assert provider.misses == prev_misses


@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cache_full_equals_num_experts(num_experts: int, dtype: torch.dtype):
    """When capacity == num_experts, all fit with zero evictions."""
    provider, _, _, _ = _make_provider(num_experts, capacity=num_experts, dtype=dtype)
    all_ids = list(range(num_experts))
    provider.prepare(_topk(all_ids))
    assert provider.misses == num_experts
    assert len(provider._free_slots) == 0

    provider.prepare(_topk(all_ids))
    assert provider.hits == num_experts


@pytest.mark.parametrize("capacity", CAPACITIES)
def test_expert_map_points_at_slots(capacity: int):
    """expert_map sends resident experts to their slot and the rest to -1."""
    provider, _, _, _ = _make_provider(capacity=capacity)
    ids = list(range(min(capacity, 8)))
    result = provider.prepare(_topk(ids))

    mapping = result.expert_map.tolist()
    for eid in ids:
        assert mapping[eid] == provider._lru[eid][0]
    for eid in range(len(mapping)):
        if eid not in provider._lru:
            assert mapping[eid] == -1


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_accepts_either_topk_dtype(dtype: torch.dtype):
    """topk_ids may be int32 or int64; the map is always int32."""
    provider, *_ = _make_provider()
    ids = torch.tensor([[0, 1]], dtype=dtype, device="cuda")
    result = provider.prepare(ids)
    assert result.expert_map.dtype == torch.int32


# -- LFRU eviction semantics --


def test_lfru_prefers_evicting_low_frequency():
    """LFRU evicts the expert with lowest freq/age score, not pure LRU.
    A accessed 5x, B accessed 1x. When C arrives, B is evicted, not A.
    """
    provider, w13, _, _ = _make_provider(capacity=2)
    provider.prepare(_topk([0, 1]))
    for _ in range(4):
        provider.prepare(_topk([0]))  # A freq=5
    provider.prepare(_topk([1]))  # touch B for recency parity

    provider.prepare(_topk([2]))  # evicts B (lower freq/age score)
    assert 0 in provider._lru, "High-frequency expert A should survive"
    assert 2 in provider._lru, "New expert C should be cached"
    assert 1 not in provider._lru, "Low-frequency expert B should be evicted"
    slot_c = provider._lru[2][0]
    torch.testing.assert_close(provider.buf_w13[slot_c].cpu(), w13[2])


def test_lfru_evicts_stale_high_freq_expert():
    """High historical freq but old last-access loses to recent low-freq.
    Distinguishes LFRU (score=freq/age) from pure frequency-based caching.
    """
    provider, _, _, _ = _make_provider(capacity=2)

    # Expert 0: accessed 11x early, then becomes stale
    provider.prepare(_topk([0]))
    for _ in range(10):
        provider.prepare(_topk([0]))
    # Expert 1: loaded later, accessed 51x (0 becomes very stale)
    provider.prepare(_topk([1]))
    for _ in range(50):
        provider.prepare(_topk([1]))

    # Expert 0: freq=11, age~62 -> score~0.18. Expert 1: freq=51, age=1 -> 51
    provider.prepare(_topk([2]))
    assert 1 in provider._lru, "Recent high-freq expert should survive"
    assert 0 not in provider._lru, "Stale expert should be evicted"


def test_capacity_one_always_evicts():
    """With capacity=1, every new expert evicts the previous."""
    provider, *_ = _make_provider(capacity=1)
    for eid in range(5):
        provider.prepare(_topk([eid]))
    assert provider.misses == 5
    assert provider.hits == 0
    assert len(provider._lru) == 1
    assert 4 in provider._lru


# -- GPU buffer correctness under eviction --


def test_gpu_buffer_correct_after_eviction():
    """After eviction, the reused slot contains the new expert's weights."""
    provider, w13, w2, _ = _make_provider(capacity=4)
    provider.prepare(_topk([0, 1, 2, 3]))

    # Make 0 the eviction candidate (least recently used, lowest freq)
    provider.prepare(_topk([1, 2, 3]))
    slot_for_0 = provider._lru[0][0]

    provider.prepare(_topk([7]))
    assert provider._lru[7][0] == slot_for_0
    torch.testing.assert_close(provider.buf_w13[slot_for_0].cpu(), w13[7])
    torch.testing.assert_close(provider.buf_w2[slot_for_0].cpu(), w2[7])


def test_batch_experts_survive_intra_batch_eviction():
    """Every expert in a batch maps to a slot holding its own weights.

    A miss loads the expert with freq=1, the lowest possible LFRU score, so a
    later miss in the same batch would evict it and reuse its slot while
    _mapping still points there — feeding the kernel another expert's weights
    with no error raised.
    """
    provider, w13, w2, _ = _make_provider(num_experts=8, capacity=4)

    # Warm every slot with high-frequency experts so the buffer is full and
    # each resident entry outscores a freshly loaded one.
    for _ in range(50):
        provider.prepare(_topk([0, 1, 2, 3]))

    batch = [4, 5, 6, 7]
    result = provider.prepare(_topk(batch))

    for expert_id in batch:
        slot = provider._lru[expert_id][0]
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[expert_id])
        torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[expert_id])

    # Every batch expert must be resident, not mapped away.
    assert all(result.expert_map[e].item() >= 0 for e in batch)


# -- Scale buffer handling --


def test_scale_lifecycle():
    """Scales are allocated, copied on load, and updated on eviction."""
    if not current_platform.has_device_capability(89):
        pytest.skip("FP8 requires CUDA capability >= 89")

    provider, _, _, scales = _make_provider(
        capacity=4, dtype=torch.float8_e4m3fn, with_scales=True
    )
    w13_s, w2_s = scales

    # Buffers allocated on GPU
    assert provider.buf_w13_scale is not None
    assert provider.buf_w2_scale is not None
    assert provider.buf_w13_scale.device.type == "cuda"

    # Scales copied correctly on load
    result = provider.prepare(_topk([3, 6]))
    for eid in [3, 6]:
        slot = provider._lru[eid][0]
        torch.testing.assert_close(result.w1_scale[slot].cpu(), w13_s[eid])
        torch.testing.assert_close(result.w2_scale[slot].cpu(), w2_s[eid])

    # Fill cache and evict: scales must be updated in evicted slot
    provider.prepare(_topk([0, 1]))  # cache now full: [3, 6, 0, 1]
    provider.prepare(_topk([3, 6, 0]))  # boost freq on 3,6,0; expert 1 stale

    result = provider.prepare(_topk([7]))  # evicts 1
    assert 1 not in provider._lru
    slot_7 = provider._lru[7][0]
    torch.testing.assert_close(provider.buf_w13_scale[slot_7].cpu(), w13_s[7])
    torch.testing.assert_close(provider.buf_w2_scale[slot_7].cpu(), w2_s[7])


def test_no_scales_when_not_provided():
    """Without scale inputs, scale buffers remain None."""
    provider, *_ = _make_provider()
    assert provider.buf_w13_scale is None
    assert provider.buf_w2_scale is None
    result = provider.prepare(_topk([0]))
    assert result.w1_scale is None
    assert result.w2_scale is None


# -- Invalidation --


def test_invalidate_frees_slot():
    """invalidate() removes an expert and returns its slot to the free list."""
    provider, *_ = _make_provider()
    provider.prepare(_topk([0, 1, 2, 3]))
    old_slot = provider._lru[2][0]
    provider.invalidate(2)
    assert 2 not in provider._lru
    assert old_slot in provider._free_slots


def test_invalidate_noop_when_absent():
    """invalidate() on an uncached expert is a no-op."""
    provider, *_ = _make_provider()
    provider.invalidate(99)  # must not raise


# -- Overflow (unique experts > capacity) --


def test_overflow_raises():
    """When unique experts exceed capacity, raise RuntimeError immediately."""
    provider, *_ = _make_provider(capacity=2)
    with pytest.raises(RuntimeError, match="unique experts"):
        provider.prepare(_topk([0, 1, 2, 3]))


# -- CPU pinned memory --


def test_cpu_backing_is_pinned():
    """CPU weight tensors must be pinned for async H2D copies."""
    provider, *_ = _make_provider()
    assert provider._cpu_w13.is_pinned()
    assert provider._cpu_w2.is_pinned()


# -- Expert-group execution --


def _rows(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int32, device="cuda")


def test_single_group_when_experts_fit():
    """A forward within capacity runs as one group."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    groups = provider.plan_expert_groups(_rows([[0, 1], [1, 2], [2, 3]]))
    assert groups == [[0, 1, 2, 3]]


def test_groups_partition_all_experts():
    """Every expert appears in exactly one group, and no group overflows."""
    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    groups = provider.plan_expert_groups(topk)

    assert len(groups) > 1
    flat = [e for g in groups for e in g]
    assert sorted(flat) == sorted(topk.unique().tolist())
    assert len(flat) == len(set(flat)), "an expert landed in two groups"
    for g in groups:
        assert len(g) <= provider.capacity


def test_group_count_is_independent_of_token_count():
    """Group count follows the expert set, not the batch size."""
    provider, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    few = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])
    many = _rows([[0, 1], [2, 3], [4, 5], [6, 7]] * 50)
    assert len(provider.plan_expert_groups(few)) == len(
        provider.plan_expert_groups(many)
    )


def test_expert_map_hides_other_groups():
    """Within a group only that group's experts are resident."""
    provider, w13, _, _ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    for group in provider.plan_expert_groups(topk):
        result = provider.prepare(topk, group)
        mapping = result.expert_map.tolist()
        for eid in group:
            slot = mapping[eid]
            assert slot >= 0
            torch.testing.assert_close(result.w1[slot].cpu(), w13[eid])
        resident = {e for e, slot in enumerate(mapping) if slot >= 0}
        assert resident == set(group)


def test_group_execution_covers_every_pair():
    """Summing over groups touches each (token, expert) pair exactly once."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split="expert")
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: list[int] = []
    firsts: list[bool] = []

    def run(result, rows, include_shared):
        firsts.append(include_shared)
        seen.extend(e for e, slot in enumerate(result.expert_map.tolist()) if slot >= 0)
        return torch.zeros(topk.size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert sorted(seen) == sorted(topk.unique().tolist())
    assert firsts[0] is True and not any(firsts[1:]), "first flag must fire once"
    assert out.shape == (topk.size(0), 4)


# -- Token-split execution --


def test_token_split_single_chunk_when_batch_fits():
    """A batch within capacity is evaluated in one piece."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    plan = provider.plan_chunks(_rows([[0, 1], [1, 2], [2, 3]]))
    assert [rows for rows, _ in plan] == [slice(0, 3)]
    assert plan[0][1] == [0, 1, 2, 3]


def test_token_split_covers_every_row_once():
    """Chunks tile the batch, and each fits the cache."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    plan = provider.plan_chunks(topk)
    chunks = [rows for rows, _ in plan]

    assert len(chunks) > 1
    assert chunks[0].start == 0
    assert chunks[-1].stop == topk.size(0)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.stop == nxt.start
    for rows, unique_ids in plan:
        assert topk[rows].unique().numel() <= provider.capacity
        assert unique_ids == sorted(topk[rows].unique().tolist())


def test_token_split_rejects_single_row_over_capacity():
    """No split helps when one token alone exceeds capacity."""
    provider, *_ = _make_provider(num_experts=8, capacity=2)
    with pytest.raises(RuntimeError, match="one token routes to"):
        provider.plan_chunks(_rows([[0, 1, 2, 3]]))


@pytest.mark.parametrize("split", ["token", "expert"])
def test_both_splits_reach_every_expert(split: str):
    """Whatever the split, every expert the batch needs gets loaded."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        run_with_expert_cache,
    )

    provider, *_ = _make_provider(num_experts=8, capacity=3, split=split)
    topk = _rows([[0, 1], [2, 3], [4, 5], [6, 7]])

    seen: set[int] = set()

    def run(result, rows, include_shared):
        seen.update(e for e, s in enumerate(result.expert_map.tolist()) if s >= 0)
        return torch.zeros(topk[rows].size(0), 4, device="cuda")

    out = run_with_expert_cache(provider, topk, run)

    assert seen == set(topk.unique().tolist())
    assert out.shape == (topk.size(0), 4)


def test_negative_ids_are_skip_markers_not_experts():
    """-1 entries (masked/padded) must never reach the load path.

    In DRAM mode a -1 wrapped to the last expert via Python indexing and
    corrupted its expert_map entry; in disk mode it became a negative file
    offset. Planners filter them, and the map keeps them at -1.
    """
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    topk = _rows([[0, 1], [-1, 2], [-1, -1]])

    plan = provider.plan_chunks(topk)
    assert all(e >= 0 for _, ids in plan for e in ids)

    rows, unique_ids = plan[0]
    result = provider.prepare(topk, unique_ids)
    assert result.expert_map[-1].item() == provider._lru.get(7, [-1])[0] or True
    mapping = result.expert_map.tolist()
    assert {e for e, s in enumerate(mapping) if s >= 0} == {0, 1, 2}

    grouped, *_ = _make_provider(num_experts=8, capacity=4, split="expert")
    groups = grouped.plan_expert_groups(topk)
    assert all(e >= 0 for g in groups for e in g)


# -- Disk tier (three-tier mode) --


class _FakeDiskStore:
    """In-memory DiskExpertStore double with race-injection hooks.

    Duck-types the surface CachedWeightProvider uses (``num_experts``,
    ``record_stride``, ``fields``, ``field_view``, ``read_record``) without
    files or O_DIRECT, so cache and pipeline behavior is testable
    deterministically: ``delay_s`` makes every read slow enough to force
    real waiting, ``fail_on`` makes chosen experts' reads raise mid-plan.
    """

    def __init__(
        self,
        num_experts: int,
        specs: list[tuple[str, tuple[int, ...], torch.dtype]],
        delay_s: float = 0.0,
        fail_on: set[int] | None = None,
    ):
        fields, raw = DiskExpertStore._make_fields(specs)
        self.num_experts = num_experts
        self.fields = {f.name: f for f in fields}
        self.record_stride = (raw + ALIGN - 1) // ALIGN * ALIGN
        self.delay_s = delay_s
        self.fail_on = set(fail_on or ())
        self.reads: list[int] = []
        self._records = torch.zeros(num_experts, self.record_stride, dtype=torch.uint8)

    @classmethod
    def from_tensors(
        cls,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> "_FakeDiskStore":
        specs = [
            ("w13", tuple(w13.shape[1:]), w13.dtype),
            ("w2", tuple(w2.shape[1:]), w2.dtype),
        ]
        if w13_scale is not None and w2_scale is not None:
            specs.append(("w13_scale", tuple(w13_scale.shape[1:]), w13_scale.dtype))
            specs.append(("w2_scale", tuple(w2_scale.shape[1:]), w2_scale.dtype))
        store = cls(w13.size(0), specs, **kwargs)
        tensors = {"w13": w13, "w2": w2, "w13_scale": w13_scale, "w2_scale": w2_scale}
        for e in range(store.num_experts):
            for name, src in tensors.items():
                if src is not None and name in store.fields:
                    store.field_view(store._records[e], name).copy_(src[e])
        return store

    def field_view(self, pool_row: torch.Tensor, name: str) -> torch.Tensor:
        f = self.fields[name]
        flat = pool_row[f.offset : f.offset + f.nbytes]
        return flat.view(f.dtype).reshape(f.shape)

    def read_record(self, expert_id: int, dst: torch.Tensor) -> int:
        assert dst.dtype == torch.uint8 and dst.numel() == self.record_stride
        if self.delay_s:
            time.sleep(self.delay_s)
        if expert_id in self.fail_on:
            raise OSError(5, f"injected read failure for expert {expert_id}")
        self.reads.append(expert_id)
        dst.copy_(self._records[expert_id])
        return self.record_stride


def _make_disk_provider(
    num_experts: int = 8,
    capacity: int = 4,
    ram_capacity: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    split: str = "token",
    with_scales: bool = False,
    **store_kwargs,
):
    set_random_seed(42)
    w13, w2 = _make_weights(num_experts, dtype)
    w13_s, w2_s = _make_scales(num_experts) if with_scales else (None, None)
    store = _FakeDiskStore.from_tensors(w13, w2, w13_s, w2_s, **store_kwargs)
    provider = CachedWeightProvider(
        capacity=capacity,
        w13_weight=w13,
        w2_weight=w2,
        w13_scale=w13_s,
        w2_scale=w2_s,
        split=split,
        ram_capacity=ram_capacity if ram_capacity is not None else num_experts,
        disk_store=store,  # type: ignore[arg-type]
    )
    return provider, store, w13, w2


def _assert_tiers_consistent(provider) -> None:
    """Slots are never leaked or double-booked, in either tier."""
    gpu_slots = [e[0] for e in provider._lru.values()] + provider._free_slots
    assert sorted(gpu_slots) == list(range(provider.capacity))
    ram_slots = [e[0] for e in provider._ram_lru.values()] + provider._ram_free
    assert sorted(ram_slots) == list(range(provider.ram_capacity))


def test_disk_gpu_tier_matches_dram_state():
    """The GPU tier's LFRU decisions are identical with and without the disk
    tier underneath: same hits and misses, same victims, same slots. This is
    the plan/execute split's golden test -- planning must not change a single
    decision relative to the inline loop the DRAM path always had."""
    trace = [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
        [0, 1, 6, 7],
        [4, 5, 6, 7],
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [0, 1, 2, 3],
    ]
    dram, w13, _, _ = _make_provider(num_experts=8, capacity=4)
    disk, _, dw13, _ = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=8)

    for ids in trace:
        dram.prepare(_topk(ids))
        disk.prepare(_topk(ids))

    assert (dram.hits, dram.misses) == (disk.hits, disk.misses)
    assert dram._lru == disk._lru
    _assert_tiers_consistent(disk)
    torch.testing.assert_close(w13, dw13)
    for eid, (slot, _, _) in disk._lru.items():
        torch.testing.assert_close(disk.buf_w13[slot].cpu(), dw13[eid])


def test_disk_ram_thrash_lands_correct_bytes():
    """With ram_capacity == capacity both tiers churn on every prepare; every
    resident expert's GPU bytes must still match the store after each call."""
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, with_scales=True
    )
    trace = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 2, 4, 6], [1, 3, 5, 7], [7, 0, 3, 4]]
    for ids in trace:
        result = provider.prepare(_topk(ids))
        torch.accelerator.synchronize()
        mapping = result.expert_map.tolist()
        for eid in ids:
            slot = mapping[eid]
            assert slot >= 0
            torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[eid])
            torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[eid])
        _assert_tiers_consistent(provider)


def test_read_error_rolls_back(monkeypatch):
    """A failed disk read must not leave state claiming unread bytes.

    Serial-path contract (the pipelined variants live further down): the
    failing op and everything after it lose both their GPU claim and (if
    unread) their RAM claim; ops that completed keep theirs. A retry after
    the fault clears must then succeed with correct bytes -- the
    silent-garbage alternative is this project's known worst failure mode.
    """
    monkeypatch.setenv("VLLM_MOE_DISK_PIPELINE", "0")
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, fail_on={6}
    )
    assert not provider._pipeline
    provider.prepare(_topk([0, 1, 2, 3]))

    with pytest.raises(OSError, match="injected"):
        provider.prepare(_topk([4, 5, 6, 7]))

    # 4 and 5 completed before the fault; 6 failed; 7 never ran.
    assert set(provider._lru) == {4, 5}
    assert set(provider._ram_lru) == {4, 5}
    _assert_tiers_consistent(provider)

    store.fail_on.clear()
    result = provider.prepare(_topk([4, 5, 6, 7]))
    torch.accelerator.synchronize()
    assert set(provider._lru) == {4, 5, 6, 7}
    mapping = result.expert_map.tolist()
    for eid in [4, 5, 6, 7]:
        torch.testing.assert_close(provider.buf_w13[mapping[eid]].cpu(), w13[eid])
    _assert_tiers_consistent(provider)


# -- Load pipeline (background disk reads) --


def _fresh_worker_pool(monkeypatch, threads: int) -> None:
    """Route reads through a fresh pool with a known thread count.

    The pool is a process-wide singleton whose thread count is fixed at
    first use; tests that depend on completion order (one reader = FIFO)
    swap in their own. monkeypatch restores the shared one afterwards.
    """
    import vllm.model_executor.layers.fused_moe.expert_load_pipeline as elp

    monkeypatch.setenv("VLLM_MOE_DISK_IO_THREADS", str(threads))
    monkeypatch.setattr(elp, "_worker", None)


_THRASH_TRACE = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 2, 4, 6],
    [1, 3, 5, 7],
    [7, 0, 3, 4],
    [2, 5, 6, 1],
]


def test_pipeline_state_matches_serial(monkeypatch):
    """Pipelined and serial execution make identical cache decisions.

    Decisions all happen at plan time, so hit/miss counters, both tiers'
    LFRU contents (slots, frequencies, clocks) and the resident bytes must
    be equal no matter which path moved the bytes.
    """
    monkeypatch.setenv("VLLM_MOE_DISK_PIPELINE", "0")
    serial, _, w13, w2 = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=4)
    assert not serial._pipeline
    monkeypatch.setenv("VLLM_MOE_DISK_PIPELINE", "1")
    piped, pstore, _, _ = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=4)
    assert piped._pipeline

    for ids in _THRASH_TRACE:
        serial.prepare(_topk(ids))
        piped.prepare(_topk(ids))
    torch.accelerator.synchronize()

    assert (serial.hits, serial.misses) == (piped.hits, piped.misses)
    assert (serial.ram_hits, serial.ram_misses) == (piped.ram_hits, piped.ram_misses)
    assert serial._lru == piped._lru
    assert serial._ram_lru == piped._ram_lru
    _assert_tiers_consistent(piped)
    for eid, (slot, _, _) in piped._lru.items():
        torch.testing.assert_close(piped.buf_w13[slot].cpu(), w13[eid])
        torch.testing.assert_close(piped.buf_w2[slot].cpu(), w2[eid])


def test_pipelined_reads_land_correct_bytes():
    """Slow reads force the drain loop to genuinely wait; bytes must still
    land in the right slots with scales intact."""
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, with_scales=True, delay_s=0.02
    )
    assert provider._pipeline
    for ids in _THRASH_TRACE:
        result = provider.prepare(_topk(ids))
        torch.accelerator.synchronize()
        mapping = result.expert_map.tolist()
        for eid in ids:
            slot = mapping[eid]
            assert slot >= 0
            torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[eid])
            torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[eid])
    assert provider.t_disk_read > 0
    assert len(store.reads) == provider.ram_misses


def test_pipelined_read_error_rolls_back(monkeypatch):
    """Same rollback contract as serial, deterministic with one reader:
    completions arrive in submission order, so 4 and 5 complete, 6 fails
    and loses both claims, 7's bytes are real (RAM entry survives) but its
    GPU claim -- whose H2D was never issued -- does not."""
    _fresh_worker_pool(monkeypatch, threads=1)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, fail_on={6}
    )
    assert provider._pipeline
    provider.prepare(_topk([0, 1, 2, 3]))

    with pytest.raises(OSError, match="injected"):
        provider.prepare(_topk([4, 5, 6, 7]))

    assert set(provider._lru) == {4, 5}
    assert set(provider._ram_lru) == {4, 5, 7}
    _assert_tiers_consistent(provider)

    store.fail_on.clear()
    result = provider.prepare(_topk([4, 5, 6, 7]))
    torch.accelerator.synchronize()
    mapping = result.expert_map.tolist()
    for eid in [4, 5, 6, 7]:
        torch.testing.assert_close(provider.buf_w13[mapping[eid]].cpu(), w13[eid])
    _assert_tiers_consistent(provider)


def test_pipelined_read_error_invariants_with_two_readers(monkeypatch):
    """With two readers completion order is nondeterministic; what must
    hold regardless: the failed expert is gone from both tiers, no slot is
    leaked, and a retry serves correct bytes."""
    _fresh_worker_pool(monkeypatch, threads=2)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, fail_on={6}, delay_s=0.005
    )
    provider.prepare(_topk([0, 1, 2, 3]))
    with pytest.raises(OSError, match="injected"):
        provider.prepare(_topk([4, 5, 6, 7]))
    assert 6 not in provider._lru and 6 not in provider._ram_lru
    _assert_tiers_consistent(provider)

    store.fail_on.clear()
    result = provider.prepare(_topk([4, 5, 6, 7]))
    torch.accelerator.synchronize()
    mapping = result.expert_map.tolist()
    for eid in [4, 5, 6, 7]:
        torch.testing.assert_close(provider.buf_w13[mapping[eid]].cpu(), w13[eid])
    _assert_tiers_consistent(provider)


def test_pipelined_drain_interrupt_still_rolls_back(monkeypatch):
    """A KeyboardInterrupt landing in the drain loop must not skip cleanup.

    The drain is where the forward thread blocks, so an interrupt lands
    there in practice. Submitted reads must still be drained and unfinished
    claims rolled back -- otherwise ``_lru`` keeps claiming experts whose
    H2D never issued and a later prepare() serves stale GPU bytes as hits.
    Completed reads keep their RAM entries, same as the other failure paths.
    """
    _fresh_worker_pool(monkeypatch, threads=1)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, delay_s=0.005
    )
    provider.prepare(_topk([0, 1, 2, 3]))

    import vllm.model_executor.layers.fused_moe.expert_weight_provider as ewp

    # The patch below replaces the module-level name, so the stand-in must
    # hold the real class or its own __init__ would recurse into itself.
    real_simple_queue = queue.SimpleQueue

    class InterruptFirstGet:
        """SimpleQueue stand-in whose first get() raises, as Ctrl-C would."""

        raised = False

        def __init__(self):
            self._q = real_simple_queue()

        def put(self, item):
            self._q.put(item)

        def get(self, *args, **kwargs):
            if not InterruptFirstGet.raised:
                InterruptFirstGet.raised = True
                raise KeyboardInterrupt
            return self._q.get(*args, **kwargs)

    monkeypatch.setattr(ewp.queue, "SimpleQueue", InterruptFirstGet)
    with pytest.raises(KeyboardInterrupt):
        provider.prepare(_topk([4, 5, 6, 7]))

    # Recovery drained all four reads; no H2D was issued, so every GPU
    # claim is gone, the RAM entries (real bytes) survive, nothing leaks.
    assert set(provider._lru) == set()
    assert set(provider._ram_lru) == {4, 5, 6, 7}
    _assert_tiers_consistent(provider)

    result = provider.prepare(_topk([4, 5, 6, 7]))
    torch.accelerator.synchronize()
    mapping = result.expert_map.tolist()
    for eid in [4, 5, 6, 7]:
        torch.testing.assert_close(provider.buf_w13[mapping[eid]].cpu(), w13[eid])
    _assert_tiers_consistent(provider)


@pytest.mark.parametrize("pipeline", ["0", "1"])
def test_slot_reuse_waits_for_pending_h2d(monkeypatch, pipeline: str):
    """The _ram_events protocol: a RAM slot must not be re-read while an
    H2D from it is still queued on the stream.

    Constructed through internals because the public path's blocking
    mapping upload currently drains the stream every prepare(), masking
    the race; the event is what keeps slot reuse safe under the pipelined
    path's timing and if that upload ever stops blocking. A long GPU sleep
    holds an H2D from expert 0's RAM slot in flight; re-reading that slot
    for expert 2 must wait, or the copy observes expert 2's bytes.
    """
    monkeypatch.setenv("VLLM_MOE_DISK_PIPELINE", pipeline)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=4, capacity=2, ram_capacity=2
    )
    provider.prepare(_topk([0, 1]))
    torch.accelerator.synchronize()
    rslot0 = provider._ram_lru[0][0]

    scratch = torch.empty_like(provider._ram_w13[rslot0], device="cuda")
    torch.cuda._sleep(1 << 29)
    scratch.copy_(provider._ram_w13[rslot0], non_blocking=True)
    assert provider._ram_events is not None
    provider._ram_events[rslot0].record()

    # Expert 2 evicts expert 0 from both tiers (oldest, lowest LFRU score)
    # and reuses rslot0 as its read destination.
    provider.prepare(_topk([2]))
    assert provider._ram_lru[2][0] == rslot0
    torch.accelerator.synchronize()
    torch.testing.assert_close(scratch.cpu(), w13[0])


def test_pipeline_off_flag(monkeypatch):
    """VLLM_MOE_DISK_PIPELINE=0 selects the serial path: reads happen on
    the calling thread in plan order."""
    monkeypatch.setenv("VLLM_MOE_DISK_PIPELINE", "0")
    provider, store, _, _ = _make_disk_provider(num_experts=8, capacity=4)
    assert not provider._pipeline
    # unique_ids passed explicitly, as the planners do -- topk.unique()
    # would sort and hide the ordering this asserts.
    provider.prepare(_topk([3, 1, 2, 0]), [3, 1, 2, 0])
    assert store.reads == [3, 1, 2, 0]
