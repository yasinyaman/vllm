# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the drained-barrier KV block pool resize.

These are pure bookkeeping tests -- no CUDA, no worker. The physical KV
tensor realloc that must accompany a resize is covered separately.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.kv_pool_resize import (
    assert_drained,
    cached_block_ids,
    is_drained,
    pinned_block_ids,
    resize_block_pool,
)

BLOCK_SIZE = 16


def make_pool(n: int, cached_ids: tuple[int, ...] = ()) -> BlockPool:
    """A drained pool of ``n`` blocks, with ``cached_ids`` in the prefix cache.

    Cached blocks stay in the free queue with ref_cnt 0 -- the "cached but
    evictable" state that makes a naive resize destructive.
    """
    pool = BlockPool(
        num_gpu_blocks=n, enable_caching=True, hash_block_size=BLOCK_SIZE
    )
    for bid in cached_ids:
        assert bid != 0, "block 0 is the null block"
        pool._insert_block_hash(
            make_block_hash_with_group_id(BlockHash(f"h{bid}".encode()), 0),
            pool.blocks[bid],
            num_tokens=BLOCK_SIZE,
        )
    return pool


def free_order(pool: BlockPool) -> list[int]:
    return [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]


def cached_ids_in(pool: BlockPool) -> set[int]:
    return cached_block_ids(pool)


def walk_links(pool: BlockPool) -> list[int]:
    """Walk the free list by hand, so a corrupted chain shows up."""
    q = pool.free_block_queue
    out = []
    node = q.fake_free_list_head.next_free_block
    while node is not None and node is not q.fake_free_list_tail:
        out.append(node.block_id)
        node = node.next_free_block
        assert len(out) <= pool.num_gpu_blocks + 4, "cycle in the free list"
    assert node is q.fake_free_list_tail, "free list did not end at its sentinel"
    return out


# ---------------------------------------------------------------------------
# preconditions
# ---------------------------------------------------------------------------


def test_fresh_pool_is_drained():
    pool = make_pool(32)
    assert is_drained(pool)
    assert_drained(pool)


def test_refuses_while_blocks_are_held():
    pool = make_pool(32)
    pool.get_new_blocks(3)
    assert not is_drained(pool)
    with pytest.raises(RuntimeError, match="drained"):
        resize_block_pool(pool, 16)


def test_refuses_degenerate_size():
    pool = make_pool(32)
    with pytest.raises(ValueError, match=">= 2"):
        resize_block_pool(pool, 1)


def test_same_size_is_a_noop():
    pool = make_pool(32, cached_ids=(3, 5))
    report = resize_block_pool(pool, 32)
    assert not report.changed
    assert pool.num_gpu_blocks == 32
    assert cached_ids_in(pool) == {3, 5}


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------


def test_shrink_updates_structure():
    pool = make_pool(64)
    report = resize_block_pool(pool, 20)
    assert report.changed
    assert report.old_blocks == 64 and report.new_blocks == 20
    assert pool.num_gpu_blocks == 20
    assert len(pool.blocks) == 20
    assert pool.free_block_queue.num_free_blocks == 19  # null is held
    assert [b.block_id for b in pool.blocks] == list(range(20))


def test_shrink_keeps_null_block_identity():
    """Single-type managers cache the null block object at construction."""
    pool = make_pool(64)
    null = pool.null_block
    resize_block_pool(pool, 20)
    assert pool.null_block is null
    assert pool.blocks[0] is null
    assert null.is_null
    assert 0 not in free_order(pool)


def test_shrink_preserves_surviving_prefix_cache_entries():
    """The whole point: a resize is not a hidden reset_prefix_cache()."""
    pool = make_pool(64, cached_ids=(2, 5, 9, 40, 55))
    report = resize_block_pool(pool, 20)
    assert cached_ids_in(pool) == {2, 5, 9}
    assert report.preserved_cached == 3
    assert report.dropped_cached == 2


def test_shrink_keeps_cached_blocks_reachable_by_hash():
    pool = make_pool(64, cached_ids=(7,))
    key = make_block_hash_with_group_id(BlockHash(b"h7"), 0)
    before = pool.cached_block_hash_to_block.get_one_block(key)
    resize_block_pool(pool, 20)
    after = pool.cached_block_hash_to_block.get_one_block(key)
    assert after is before, "cache entry must still point at the same block object"
    assert after is pool.blocks[7]


def test_shrink_drops_hashes_of_removed_blocks():
    pool = make_pool(64, cached_ids=(50,))
    key = make_block_hash_with_group_id(BlockHash(b"h50"), 0)
    assert pool.cached_block_hash_to_block.get_one_block(key) is not None
    resize_block_pool(pool, 20)
    assert pool.cached_block_hash_to_block.get_one_block(key) is None
    assert 50 not in pool.cached_block_hashes_by_block


def test_shrink_preserves_eviction_order():
    """Free-queue order is the LRU policy; rebuilding by id would erase it."""
    pool = make_pool(64)
    # Churn the queue so its order is no longer id order.
    taken = pool.get_new_blocks(10)
    pool.free_blocks(reversed(taken))
    assert is_drained(pool)
    before = free_order(pool)
    assert before != sorted(before), "test needs a non-id ordering to be meaningful"

    resize_block_pool(pool, 30)
    after = free_order(pool)
    expected = [bid for bid in before if bid < 30]
    assert after == expected


def test_shrink_then_allocate_stays_in_range():
    pool = make_pool(64, cached_ids=(3, 40))
    resize_block_pool(pool, 20)
    blocks = pool.get_new_blocks(19)
    assert all(b.block_id < 20 for b in blocks)
    with pytest.raises(ValueError):
        pool.get_new_blocks(1)  # pool is now empty


# ---------------------------------------------------------------------------
# grow
# ---------------------------------------------------------------------------


def test_grow_updates_structure():
    pool = make_pool(16)
    report = resize_block_pool(pool, 40)
    assert report.changed
    assert pool.num_gpu_blocks == 40
    assert len(pool.blocks) == 40
    assert pool.free_block_queue.num_free_blocks == 39
    assert [b.block_id for b in pool.blocks] == list(range(40))


def test_grow_hands_out_fresh_blocks_first():
    """Fresh blocks cache nothing, so consuming them evicts nothing warm."""
    pool = make_pool(16, cached_ids=(3, 7))
    resize_block_pool(pool, 40)
    head = free_order(pool)[: 40 - 16]
    assert set(head) == set(range(16, 40))
    # The warm blocks are still cached after the new ones are consumed.
    pool.get_new_blocks(24)
    assert cached_ids_in(pool) == {3, 7}


def test_grow_preserves_prefix_cache():
    pool = make_pool(16, cached_ids=(3, 7, 11))
    report = resize_block_pool(pool, 64)
    assert cached_ids_in(pool) == {3, 7, 11}
    assert report.dropped_cached == 0
    assert report.preserved_cached == 3


# ---------------------------------------------------------------------------
# round trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes", [(64, 16, 64), (16, 64, 16), (64, 63, 64), (32, 2, 32)]
)
def test_round_trip_keeps_the_pool_consistent(sizes):
    start, *rest = sizes
    pool = make_pool(start, cached_ids=(1,))
    for target in rest:
        resize_block_pool(pool, target)
        assert pool.num_gpu_blocks == target
        assert len(pool.blocks) == target
        assert pool.free_block_queue.num_free_blocks == target - 1
        assert pool.blocks[0] is pool.null_block
        assert [b.block_id for b in pool.blocks] == list(range(target))
        assert sorted(free_order(pool)) == list(range(1, target))


def test_repeated_resize_does_not_leak_free_list_entries():
    pool = make_pool(48)
    for target in (12, 30, 8, 48, 20, 48):
        resize_block_pool(pool, target)
        walked = free_order(pool)
        assert len(walked) == pool.free_block_queue.num_free_blocks
        assert len(set(walked)) == len(walked), "duplicate block in the free list"


def test_usage_accounting_survives_resize():
    pool = make_pool(64)
    resize_block_pool(pool, 20)
    assert pool.get_usage() == pytest.approx(0.0)
    pool.get_new_blocks(9)
    # get_usage discounts the null block from the total.
    assert pool.get_usage() == pytest.approx(9 / 19)


# ---------------------------------------------------------------------------
# linked-list integrity  (a dropped block that keeps its links poisons any
# cursor held across steps -- SimpleCPUOffloadManager keeps exactly one)
# ---------------------------------------------------------------------------


def test_shrink_unlinks_dropped_blocks():
    pool = make_pool(64)
    dropped = list(pool.blocks[20:])
    resize_block_pool(pool, 20)
    for b in dropped:
        assert b.prev_free_block is None and b.next_free_block is None, (
            f"block {b.block_id} still points into the free list after being "
            "dropped; a cursor parked on it would walk into an orphan"
        )


def test_shrink_keeps_the_queue_object_and_its_sentinels():
    """External holders keep a reference to the queue and its tail sentinel."""
    pool = make_pool(64)
    queue = pool.free_block_queue
    head, tail = queue.fake_free_list_head, queue.fake_free_list_tail
    resize_block_pool(pool, 20)
    assert pool.free_block_queue is queue
    assert pool.free_block_queue.fake_free_list_head is head
    assert pool.free_block_queue.fake_free_list_tail is tail


def test_a_cursor_parked_on_the_tail_still_terminates():
    """The exact shape that killed lazy CPU offload: cursor on the last free
    block, that block dropped by the shrink."""
    pool = make_pool(64)
    cursor = pool.free_block_queue.get_all_free_blocks()[-1]
    assert cursor.block_id >= 20, "test needs the cursor inside the doomed range"
    resize_block_pool(pool, 20)
    assert cursor.next_free_block is None
    assert list(pool.free_block_queue.iter_blocks_after(None)) == [
        pool.blocks[i] for i in walk_links(pool)
    ]


def test_links_stay_consistent_after_churn():
    pool = make_pool(48)
    for target in (12, 30, 8, 48, 20, 48):
        resize_block_pool(pool, target)
        assert walk_links(pool) == free_order(pool)


# ---------------------------------------------------------------------------
# permanently-claimed (attention sink) blocks
# ---------------------------------------------------------------------------


def claim_sink(pool: BlockPool, n: int) -> list:
    """Mimic SinkFullAttentionManager: pop blocks out of the queue for good,
    without touching ref_cnt."""
    return pool.free_block_queue.popleft_n(n)


def test_sink_blocks_do_not_count_as_busy():
    pool = make_pool(64)
    sinks = claim_sink(pool, 8)
    assert pinned_block_ids(pool) == {b.block_id for b in sinks}
    assert is_drained(pool), "sink blocks are claimed, not referenced"
    assert_drained(pool)


def test_resize_works_with_sink_blocks_claimed():
    pool = make_pool(64)
    claim_sink(pool, 8)
    report = resize_block_pool(pool, 40)
    assert report.changed and pool.num_gpu_blocks == 40


def test_shrink_refuses_to_drop_a_claimed_block():
    pool = make_pool(64)
    # Park a claimed block high in the pool by draining the queue down to it.
    pool.free_block_queue.popleft_n(50)
    claimed = pool.free_block_queue.popleft_n(1)[0]
    assert claimed.block_id >= 20
    with pytest.raises(ValueError, match="permanently claimed"):
        resize_block_pool(pool, 20)


def test_referenced_blocks_still_block_a_resize():
    pool = make_pool(64)
    pool.get_new_blocks(3)
    assert not is_drained(pool)
    with pytest.raises(RuntimeError, match="drained"):
        resize_block_pool(pool, 20)


# ---------------------------------------------------------------------------
# admission headroom
# ---------------------------------------------------------------------------


class _FakeKVCacheManager:
    def __init__(self, watermark: float, num_blocks: int) -> None:
        self.watermark = watermark
        self.watermark_blocks = int(watermark * num_blocks)


def test_watermark_is_rescaled_with_the_pool():
    """Frozen headroom turns a 1% reserve into a 15% admission tax."""
    pool = make_pool(600)
    mgr = _FakeKVCacheManager(0.01, 600)
    assert mgr.watermark_blocks == 6
    resize_block_pool(pool, 100, kv_cache_manager=mgr)
    assert mgr.watermark_blocks == 1
    resize_block_pool(pool, 600, kv_cache_manager=mgr)
    assert mgr.watermark_blocks == 6


def test_resize_without_a_manager_is_still_fine():
    pool = make_pool(64)
    assert resize_block_pool(pool, 20).changed


def test_real_kv_cache_manager_exposes_the_watermark_fraction():
    """kv_pool_resize rescales via this attribute; it must exist."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    assert "self.watermark = watermark" in __import__("inspect").getsource(
        KVCacheManager.__init__
    )


# ---------------------------------------------------------------------------
# report accounting
# ---------------------------------------------------------------------------


def test_report_counts_blocks_not_hashes():
    """One block can carry several hashes; the report is about blocks."""
    pool = make_pool(64, cached_ids=(5,))
    pool._insert_block_hash(
        make_block_hash_with_group_id(BlockHash(b"second"), 0),
        pool.blocks[5],
        num_tokens=BLOCK_SIZE,
    )
    assert len(pool.cached_block_hash_to_block) == 2
    report = resize_block_pool(pool, 20)
    assert report.preserved_cached == 1
    assert report.dropped_cached == 0
