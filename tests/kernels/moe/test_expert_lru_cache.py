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
    quantize_rowwise_fp8,
)
from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
    ExpertWeightResult,
    run_with_expert_cache,
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
        quantize_fp8: bool = False,
        **kwargs,
    ) -> "_FakeDiskStore":
        if quantize_fp8:
            # Mirrors DiskExpertStore.build's fp8e4m3-row layout: quantized
            # weights plus one fp32 scale per output channel.
            assert w13_scale is None and w2_scale is None
            w13, w13_qs = quantize_rowwise_fp8(w13)
            w2, w2_qs = quantize_rowwise_fp8(w2)
        specs = [
            ("w13", tuple(w13.shape[1:]), w13.dtype),
            ("w2", tuple(w2.shape[1:]), w2.dtype),
        ]
        if quantize_fp8:
            specs.append(("w13_qs", tuple(w13_qs.shape[1:]), w13_qs.dtype))
            specs.append(("w2_qs", tuple(w2_qs.shape[1:]), w2_qs.dtype))
        if w13_scale is not None and w2_scale is not None:
            specs.append(("w13_scale", tuple(w13_scale.shape[1:]), w13_scale.dtype))
            specs.append(("w2_scale", tuple(w2_scale.shape[1:]), w2_scale.dtype))
        store = cls(w13.size(0), specs, **kwargs)
        tensors = {"w13": w13, "w2": w2, "w13_scale": w13_scale, "w2_scale": w2_scale}
        if quantize_fp8:
            tensors.update({"w13_qs": w13_qs, "w2_qs": w2_qs})
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
    # from_tensors may quantize its copy; the provider still sees the
    # model-dtype tensors, which is what sizes its buffers.
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
    ram_slots = (
        [e[0] for e in provider._ram_lru.values()]
        + provider._ram_free
        + [slot for _, slot in provider._ram_pending.values()]
    )
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


def test_prefetch_overlaps_and_matches_serial_bytes(monkeypatch):
    """With VLLM_MOE_DISK_PREFETCH=1, a split forward's later groups adopt
    reads started under the previous group's compute; every expert is read
    exactly once and the served bytes match the store."""
    monkeypatch.setenv("VLLM_MOE_DISK_PREFETCH", "1")
    _fresh_worker_pool(monkeypatch, threads=2)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=2, ram_capacity=8, split="expert", delay_s=0.002
    )
    assert provider._prefetch

    topk = _topk([0, 1, 2, 3, 4, 5, 6, 7])
    out = run_with_expert_cache(
        provider,
        topk,
        lambda result, rows, include_shared: torch.zeros(1, device="cuda"),
    )
    torch.accelerator.synchronize()
    provider._drain_ready_prefetches()
    assert sorted(store.reads) == list(range(8))
    assert len(store.reads) == 8, "each expert must be read exactly once"
    _assert_tiers_consistent(provider)
    for eid, (slot, _, _) in provider._ram_lru.items():
        torch.testing.assert_close(
            store.field_view(provider._ram_pool[slot], "w13"), w13[eid]
        )
    assert out is not None


def test_prefetch_failure_falls_back_to_fresh_read(monkeypatch):
    """A failed prefetch read must not poison anything: the adopting
    prepare() falls back to a fresh on-demand read with the serial path's
    loud semantics, and clearing the fault heals on retry."""
    monkeypatch.setenv("VLLM_MOE_DISK_PREFETCH", "1")
    _fresh_worker_pool(monkeypatch, threads=1)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=2, ram_capacity=8, fail_on={2}
    )
    provider.prepare(_topk([0, 1]))
    provider.prefetch_to_ram([2, 3], [0, 1])

    with pytest.raises(OSError, match="injected"):
        provider.prepare(_topk([2, 3]))
    _assert_tiers_consistent(provider)

    store.fail_on.clear()
    result = provider.prepare(_topk([2, 3]))
    torch.accelerator.synchronize()
    mapping = result.expert_map.tolist()
    for eid in [2, 3]:
        torch.testing.assert_close(provider.buf_w13[mapping[eid]].cpu(), w13[eid])
    _assert_tiers_consistent(provider)


def test_prefetch_requires_ram_slack(monkeypatch):
    """The prefetch flag must stay off when ram_capacity < 2x capacity --
    without slack a victim scan could find nothing safe to evict."""
    monkeypatch.setenv("VLLM_MOE_DISK_PREFETCH", "1")
    provider, *_ = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=4)
    assert not provider._prefetch
    provider.prefetch_to_ram([4, 5], [0, 1])  # must be a no-op
    assert not provider._ram_pending
    _assert_tiers_consistent(provider)


# -- FP8-quantized store (fp8e4m3-row) --


def _aligned_pinned_row(nbytes: int) -> torch.Tensor:
    backing = torch.empty(nbytes + ALIGN, dtype=torch.uint8).pin_memory()
    skip = (-backing.data_ptr()) % ALIGN
    row = backing[skip : skip + nbytes]
    row._backing = backing  # keep alive
    return row


def test_fp8_store_roundtrip(tmp_path):
    """Row-scaled FP8 records reconstruct weights within e4m3 resolution,
    halve the weight payload, and fingerprint separately from plain
    stores (a plain build at the same path must rebuild, not reuse)."""
    set_random_seed(42)
    w13, w2 = _make_weights(4, torch.bfloat16)
    path = str(tmp_path / "l0.experts")

    store = DiskExpertStore.build(path, w13, w2, quantize_fp8=True)
    assert store.is_complete
    assert store.fields["w13"].dtype == torch.float8_e4m3fn
    assert store.fields["w13"].nbytes == w13[0].nbytes // 2

    row = _aligned_pinned_row(store.record_stride)
    store.read_record(2, row)
    deq = store.field_view(row, "w13").float() * store.field_view(
        row, "w13_qs"
    ).unsqueeze(-1)
    torch.testing.assert_close(deq, w13[2].float(), rtol=0.08, atol=2e-2)

    reused = DiskExpertStore.build(path, w13, w2, quantize_fp8=True)
    assert reused.is_complete
    plain = DiskExpertStore.build(path, w13, w2)
    assert plain.fields["w13"].dtype == torch.bfloat16


def test_fp8_store_provider_serves_dequantized(tmp_path):
    """The fill path stages fp8 on the GPU and dequantizes into the bf16
    slot; served bytes must match the CPU dequant reference through
    thrash, and the RAM pool stride is the halved record."""
    set_random_seed(42)
    w13, w2 = _make_weights(8, torch.bfloat16)
    store = DiskExpertStore.build(
        str(tmp_path / "l0.experts"), w13, w2, quantize_fp8=True
    )
    provider = CachedWeightProvider(
        capacity=4,
        w13_weight=w13,
        w2_weight=w2,
        ram_capacity=8,
        disk_store=store,
    )
    assert provider._store_fp8

    q13, s13 = quantize_rowwise_fp8(w13)
    q2, s2 = quantize_rowwise_fp8(w2)
    ref13 = q13.to(torch.bfloat16) * s13.unsqueeze(-1).to(torch.bfloat16)
    ref2 = q2.to(torch.bfloat16) * s2.unsqueeze(-1).to(torch.bfloat16)

    for ids in [[0, 1, 2, 3], [4, 5, 6, 7], [1, 3, 5, 7], [0, 2, 4, 6]]:
        result = provider.prepare(_topk(ids))
        torch.accelerator.synchronize()
        mapping = result.expert_map.tolist()
        for eid in ids:
            torch.testing.assert_close(
                provider.buf_w13[mapping[eid]].cpu(), ref13[eid], rtol=0, atol=0
            )
            torch.testing.assert_close(
                provider.buf_w2[mapping[eid]].cpu(), ref2[eid], rtol=0, atol=0
            )
    _assert_tiers_consistent(provider)


def test_fp8_streaming_store_roundtrip(tmp_path):
    """Streaming builds quantize shard-at-a-time -- every shard delivers
    complete output rows, so row scales are exact per arrival. The sealed
    store reads back within e4m3 bounds and reuses under the quant
    fingerprint."""
    set_random_seed(42)
    w13, w2 = _make_weights(4, torch.bfloat16)
    specs = [
        ("w13", (2 * INTERMEDIATE, HIDDEN), torch.float8_e4m3fn),
        ("w2", (HIDDEN, INTERMEDIATE), torch.float8_e4m3fn),
        ("w13_qs", (2 * INTERMEDIATE,), torch.float32),
        ("w2_qs", (HIDDEN,), torch.float32),
    ]
    path = str(tmp_path / "l0.experts")
    store = DiskExpertStore.create_for_streaming(path, 4, specs, quant="fp8e4m3-row")
    assert not store.is_complete
    for e in range(4):
        row = torch.zeros(store.record_stride, dtype=torch.uint8)
        for shard, lo in (
            (w13[e, :INTERMEDIATE], 0),
            (w13[e, INTERMEDIATE:], INTERMEDIATE),
        ):
            q, s = quantize_rowwise_fp8(shard)
            store.field_view(row, "w13").narrow(0, lo, INTERMEDIATE).copy_(q)
            store.field_view(row, "w13_qs").narrow(0, lo, INTERMEDIATE).copy_(s)
        q, s = quantize_rowwise_fp8(w2[e])
        store.field_view(row, "w2").copy_(q)
        store.field_view(row, "w2_qs").copy_(s)
        store.write_record(e, row)
    store.finalize()
    assert store.is_complete

    reused = DiskExpertStore.create_for_streaming(path, 4, specs, quant="fp8e4m3-row")
    assert reused.is_complete

    row = _aligned_pinned_row(store.record_stride)
    store.read_record(3, row)
    deq = store.field_view(row, "w13").float() * store.field_view(
        row, "w13_qs"
    ).unsqueeze(-1)
    torch.testing.assert_close(deq, w13[3].float(), rtol=0.08, atol=2e-2)
    store.close()


def test_ram_pool_is_exactly_pinned():
    """The RAM pool must be page-locked at its exact size, not through the
    caching allocator's power-of-two buckets (a 604 MB request measured
    1027 MB of RSS there), and page-aligned for O_DIRECT."""
    provider, *_ = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=8)
    assert provider._ram_pool is not None
    assert provider._ram_pool.data_ptr() % ALIGN == 0
    assert provider._ram_pool.is_pinned()
    assert provider._ram_pool_region.registered


def test_dram_mirrors_are_exactly_pinned():
    """Full-DRAM mode's weight mirrors take the same exact-size path; the
    caching allocator's buckets waste up to ~40% on non-power-of-two
    expert tensors."""
    provider, *_ = _make_provider(num_experts=8, capacity=4)
    assert provider._cpu_w13 is not None and provider._cpu_w13.is_pinned()
    assert provider._cpu_w2 is not None and provider._cpu_w2.is_pinned()
    assert provider._cpu_w13._pinned_region.registered


def test_pin_budget_guard_refuses_absurd_locks():
    """A page-lock near the free-RAM scale must fail loudly with a sizing
    hint -- the alternative, measured the hard way, is a livelocked host."""
    from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
        _check_pin_budget,
    )

    with pytest.raises(ValueError, match="Refusing to page-lock"):
        _check_pin_budget(1 << 50)


def test_worker_pool_shutdown_drains_and_restarts(monkeypatch):
    """shutdown_disk_load_worker() lets queued reads finish -- sentinels go
    through the same FIFO, behind the jobs -- joins the readers, and the
    next prepare() restarts a fresh pool transparently."""
    import vllm.model_executor.layers.fused_moe.expert_load_pipeline as elp

    _fresh_worker_pool(monkeypatch, threads=2)
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=4, ram_capacity=4, delay_s=0.005
    )
    assert provider._ram_pool is not None
    worker = elp.get_disk_load_worker()
    done: queue.SimpleQueue = queue.SimpleQueue()
    worker.submit(store, 6, provider._ram_pool[0], done, 0)
    worker.submit(store, 7, provider._ram_pool[1], done, 1)
    threads = list(worker._threads)
    assert threads

    elp.shutdown_disk_load_worker()
    assert all(not t.is_alive() for t in threads)
    assert {done.get(timeout=1.0)[0] for _ in range(2)} == {0, 1}

    result = provider.prepare(_topk([0, 1, 2, 3]))
    torch.accelerator.synchronize()
    mapping = result.expert_map.tolist()
    for eid in [0, 1, 2, 3]:
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


# -- eviction policy seam --


def test_policy_env_selects_ewma(monkeypatch):
    """VLLM_MOE_CACHE_POLICY/DECAY reach the per-tier policy instances."""
    from vllm.model_executor.layers.fused_moe.expert_policy import (
        EWMAPolicy,
        LFRUPolicy,
    )

    provider, *_ = _make_provider(capacity=2)
    assert type(provider._gpu_policy) is LFRUPolicy

    monkeypatch.setenv("VLLM_MOE_CACHE_POLICY", "ewma")
    monkeypatch.setenv("VLLM_MOE_CACHE_DECAY", "0.99")
    provider, *_ = _make_provider(capacity=2)
    assert type(provider._gpu_policy) is EWMAPolicy
    assert provider._gpu_policy.decay == 0.99
    assert type(provider._ram_policy) is EWMAPolicy
    assert provider._ram_policy is not provider._gpu_policy


@pytest.mark.parametrize("policy", ["lfru", "ewma"])
def test_ewma_history_survives_eviction(monkeypatch, policy):
    """The one sequence where the policies must disagree.

    Expert 0 builds history, is evicted by [1, 2], returns, and then a new
    expert needs a slot. LFRU re-admitted 0 at freq=1, so 0 loses; EWMA kept
    0's statistics across the eviction, so 2 loses. Locks both the seam's
    default behavior and the property EWMA exists for.
    """
    monkeypatch.setenv("VLLM_MOE_CACHE_POLICY", policy)
    provider, *_ = _make_provider(capacity=2)
    for _ in range(5):
        provider.prepare(_topk([0]))
    provider.prepare(_topk([1, 2]))  # evicts 0
    provider.prepare(_topk([0]))  # returns; LFRU sees a stranger
    provider.prepare(_topk([2]))
    provider.prepare(_topk([1]))
    if policy == "lfru":
        assert sorted(provider._lru) == [1, 2], "LFRU forgets evicted history"
    else:
        assert sorted(provider._lru) == [0, 1], "EWMA remembers it"


def test_ewma_prior_biases_eviction():
    """A manifest prior must outweigh equal live history; empty prior is
    inert. This is the seam a Faz C manifest seeds."""
    from vllm.model_executor.layers.fused_moe.expert_policy import EWMAPolicy

    p = EWMAPolicy(8, decay=0.999)
    p.on_insert(5)
    p.on_insert(6)
    assert p.score(5, 0, 0, 0) < p.score(6, 0, 0, 0), "6 is fresher"
    p.prior = {5: 3.0}
    assert p.score(5, 0, 0, 0) > p.score(6, 0, 0, 0), "prior lifts 5 over 6"


# -- zero copy: the kernel reads the RAM pool directly --


@pytest.fixture
def zero_copy(monkeypatch):
    monkeypatch.setenv("VLLM_MOE_ZERO_COPY", "1")


def test_zero_copy_buffers_alias_the_ram_pool(zero_copy):
    """buf_w13/buf_w2 are strided views over the pinned pool, not copies.

    The kernel only requires stride(-1) == 1 of its weights, so the expert
    dimension may stride across whole records -- that is what removes the
    fill. Writing a pool row must therefore show up in the buffer.
    """
    provider, store, w13, w2 = _make_disk_provider(num_experts=8, capacity=8)
    assert provider._zero_copy
    assert provider.capacity == provider.ram_capacity
    assert provider.buf_w13.stride(-1) == 1
    assert provider.buf_w2.stride(-1) == 1
    assert provider.buf_w13.stride(0) * provider.buf_w13.element_size() == (
        store.record_stride
    )

    provider.prepare(_topk([3]))
    slot = provider._ram_lru[3][0]
    torch.accelerator.synchronize()
    torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[3])
    torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[3])
    # No GPU tier exists to fill.
    assert not provider._lru


def test_zero_copy_map_points_at_ram_slots(zero_copy):
    """expert_map indexes the RAM pool, and exposes only this group."""
    provider, *_ = _make_disk_provider(num_experts=8, capacity=8)
    provider.prepare(_topk([5, 2]))
    torch.accelerator.synchronize()
    mapping = provider._mapping.cpu()
    for eid in (5, 2):
        assert mapping[eid].item() == provider._ram_lru[eid][0]
    assert (mapping[[0, 1, 3, 4, 6, 7]] == -1).all()


def test_zero_copy_serves_correct_bytes_under_eviction(zero_copy):
    """A slot recycled to a new expert serves that expert's weights.

    The whole risk of dropping the fill is that the kernel now reads memory
    a disk read may overwrite; this drives eviction hard and checks the
    bytes each time.
    """
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=2, ram_capacity=2
    )
    for eid in [0, 1, 2, 3, 0, 4, 1, 5]:
        provider.prepare(_topk([eid]))
        slot = provider._ram_lru[eid][0]
        torch.accelerator.synchronize()
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[eid])
        torch.testing.assert_close(provider.buf_w2[slot].cpu(), w2[eid])
    _assert_tiers_consistent(provider)


def test_zero_copy_matches_fill_path_bitwise(zero_copy, monkeypatch):
    """Same routing through both paths must give the same kernel inputs.

    Zero copy changes where the weights live, never what they are; this is
    the correctness invariant the whole mode rests on.
    """
    trace = [[0, 1], [2, 3], [1, 2], [4, 5], [0, 5], [3, 4]]
    zc, _, w13, w2 = _make_disk_provider(num_experts=8, capacity=8)
    seen_zc = []
    for grp in trace:
        zc.prepare(_topk(grp))
        torch.accelerator.synchronize()
        seen_zc.append([zc.buf_w13[zc._mapping[e].item()].cpu().clone() for e in grp])

    monkeypatch.setenv("VLLM_MOE_ZERO_COPY", "0")
    fill, *_ = _make_disk_provider(num_experts=8, capacity=8)
    assert not fill._zero_copy
    for grp, want in zip(trace, seen_zc):
        fill.prepare(_topk(grp))
        torch.accelerator.synchronize()
        for eid, ref in zip(grp, want):
            got = fill.buf_w13[fill._mapping[eid].item()].cpu()
            assert torch.equal(got, ref)
            assert torch.equal(got, w13[eid])


def test_zero_copy_dequantizes_an_fp8_store(zero_copy):
    """An fp8 store feeds a bf16 pool: half the disk, kernel-readable rows.

    The pool row stops matching the store record, so this checks both halves:
    the layout diverges (bf16-sized rows, record-sized staging) and the bytes
    the kernel sees are the dequantized weights, not the fp8 ones.
    """
    provider, store, w13, w2 = _make_disk_provider(
        num_experts=8, capacity=8, quantize_fp8=True
    )
    assert provider._zc_pool_fields, "fp8 + zero copy must use a dequantizing pool"
    assert not provider._pipeline, "the dequantizing pool needs serial reads"
    assert provider._buf_w13.dtype == w13.dtype
    # Pool rows carry dequantized weights, so w2 starts one bf16 w13 in --
    # twice as far as in the fp8 record. (Both strides round to the same
    # ALIGN block at this fixture's size, so compare offsets, not strides.)
    assert provider._zc_pool_fields["w2"][0] == 2 * store.fields["w2"].offset

    provider.prepare(_topk([0, 3, 5]))
    torch.accelerator.synchronize()
    for eid in (0, 3, 5):
        # Zero copy indexes the pool by RAM slot: that is what its
        # expert_map hands the kernel.
        slot = provider._ram_lru[eid][0]
        # Row-wise fp8 is lossy by design, so compare against what the store
        # reconstructs, in the same dtype order the provider uses.
        q, qs = quantize_rowwise_fp8(w13[eid])
        want = q.to(w13.dtype) * qs.unsqueeze(-1).to(w13.dtype)
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), want)


def test_zero_copy_fp8_staging_rows_are_recycled(zero_copy):
    """More misses than staging rows must still land the right bytes.

    A row is reused only after the dequantize that read it has run; getting
    that wrong would expand a half-overwritten record into the pool.
    """
    provider, store, w13, _ = _make_disk_provider(
        num_experts=8, capacity=8, quantize_fp8=True
    )
    # Eight cold misses through a four-row ring: every row is reused twice.
    provider.prepare(_topk(list(range(8))))
    torch.accelerator.synchronize()
    for eid in range(8):
        slot = provider._ram_lru[eid][0]
        q, qs = quantize_rowwise_fp8(w13[eid])
        want = q.to(w13.dtype) * qs.unsqueeze(-1).to(w13.dtype)
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), want)


def test_zero_copy_rejects_prefetch(zero_copy, monkeypatch):
    """A prefetch writes a slot outside prepare(), where the kernel-read
    event protocol cannot see it."""
    monkeypatch.setenv("VLLM_MOE_DISK_PREFETCH", "1")
    with pytest.raises(ValueError, match="PREFETCH"):
        _make_disk_provider(num_experts=8, capacity=8, ram_capacity=16)


def test_zero_copy_tracks_readers_by_generation(zero_copy):
    """One event per prepare() covers every slot that call exposed.

    Recording an event per exposed slot would cost thousands of
    cudaEventRecord calls per token at realistic capacities -- more than the
    fill it replaces. Slots are stamped with the call that exposed them and
    wait on that call's successor event instead.
    """
    provider, *_ = _make_disk_provider(num_experts=8, capacity=4, ram_capacity=4)
    provider.prepare(_topk([0, 1]))
    gen_after_first = provider._gen
    for eid in (0, 1):
        assert provider._slot_gen[provider._ram_lru[eid][0]] == gen_after_first

    provider.prepare(_topk([2, 3]))
    # One generation per prepare(), regardless of how many slots it exposed.
    assert provider._gen == gen_after_first + 1
    # Untouched slots keep their older stamp; the new group gets the new one.
    for eid in (0, 1):
        assert provider._slot_gen[provider._ram_lru[eid][0]] == gen_after_first
    for eid in (2, 3):
        assert provider._slot_gen[provider._ram_lru[eid][0]] == provider._gen

    # The stamp is taken after the counter advances, so the event covering a
    # slot's readers is _gen_events[stamp % ring] -- not stamp + 1. Getting
    # this off by one lets a disk read overwrite a slot mid-kernel.
    slot0 = provider._ram_lru[0][0]
    covering = provider._gen_events[provider._slot_gen[slot0] % 4]
    assert covering is provider._gen_events[gen_after_first % 4]
    assert covering.query(), "the event covering slot 0's kernel must exist"


def test_zero_copy_survives_generation_ring_wraparound(zero_copy):
    """A slot untouched for longer than the event ring is still safe.

    Its recorded generation resolves to a later event in the same ring
    position, which sits further down the stream -- the wait over-waits
    rather than under-waits, so recycled bytes stay correct.
    """
    provider, _, w13, _ = _make_disk_provider(
        num_experts=16, capacity=2, ram_capacity=2
    )
    provider.prepare(_topk([0]))
    for eid in range(1, 12):  # many generations, ring is 4 deep
        provider.prepare(_topk([eid]))
        slot = provider._ram_lru[eid][0]
        torch.accelerator.synchronize()
        torch.testing.assert_close(provider.buf_w13[slot].cpu(), w13[eid])
    _assert_tiers_consistent(provider)
