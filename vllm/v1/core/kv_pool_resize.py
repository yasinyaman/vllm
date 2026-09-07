# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resize the scheduler's KV block pool in place, at a drained barrier.

This is the scheduler half of an MV-WSA split move (``vllm.v1.core.mvwsa_policy``
decides, the worker half re-sizes the physical KV tensors and the MoE expert
slot buffers). It only touches bookkeeping: the block objects, the free-list
order, and the prefix-cache index.

Three properties are load-bearing, and all three are easy to get wrong.

**Block identity survives.** ``cached_block_hash_to_block`` maps a hash to a
``KVCacheBlock`` *object*, and ``cached_block_hashes_by_block`` is keyed by
block id. Rebuilding ``blocks`` from scratch -- ``[KVCacheBlock(i) for i in
range(n)]`` -- silently orphans every entry in both, which is
``reset_prefix_cache()`` wearing a disguise. We reuse the existing objects for
every surviving id and evict, explicitly and only, the ids that go away.
On an agentic trace the prefix cache is most of the value of the KV pool; a
resize that quietly drops it makes the next turn re-prefill, which then reads
back as "KV demand was low", which shrinks the pool again.

**The free queue object survives, and so do its links.** The queue is an
intrusive doubly-linked list threaded through the blocks themselves, with
sentinel head/tail nodes. Replacing it with ``FreeKVCacheBlockQueue(survivors)``
would allocate *new* sentinels and rewrite links only on the survivors, leaving
every dropped block still pointing into the old chain -- and anything holding a
``KVCacheBlock`` cursor across steps (``SimpleCPUOffloadManager`` does) would
walk that chain into an orphaned sentinel and silently stop producing work. So
we unlink each dropped block with the queue's own O(1) ``remove()`` and keep
the object, the sentinels, and the surviving order untouched.

**Eviction order survives.** The queue is ordered by eviction priority, not by
id -- least-recently-used at the head. Rebuilding it from ``self.blocks`` would
reset that to id order and throw away the LRU information the pool spent the
whole epoch accumulating.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)

__all__ = [
    "PoolResizeReport",
    "assert_drained",
    "cached_block_ids",
    "is_drained",
    "pinned_block_ids",
    "resize_block_pool",
]


@dataclass(frozen=True)
class PoolResizeReport:
    old_blocks: int
    new_blocks: int
    #: Blocks whose prefix-cache entries were discarded because the block
    #: itself went away. Counted in blocks, not hashes: one block can carry
    #: several hashes, and several blocks can share one.
    dropped_cached: int
    #: Blocks still holding prefix-cache entries after the move.
    preserved_cached: int
    changed: bool


def _free_block_ids(block_pool: BlockPool) -> set[int]:
    return {b.block_id for b in block_pool.free_block_queue.get_all_free_blocks()}


def cached_block_ids(block_pool: BlockPool) -> set[int]:
    """Ids of blocks currently carrying at least one prefix-cache entry."""
    ids = set(block_pool.cached_block_hashes_by_block)
    ids.update(b.block_id for b in block_pool.blocks if b.block_hash is not None)
    return ids


def pinned_block_ids(block_pool: BlockPool) -> set[int]:
    """Ids held out of the free queue permanently, without a reference count.

    ``SinkFullAttentionManager`` claims its sink blocks with
    ``free_block_queue.popleft_n(...)``, which does not touch ``ref_cnt``.
    Such a block is neither free nor referenced, so any "is the pool idle"
    test written against the free-queue count is wrong on sink-attention
    models -- and a resize that drops one would hand the manager a block id
    the pool no longer owns.
    """
    free = _free_block_ids(block_pool)
    return {
        b.block_id
        for b in block_pool.blocks
        if not b.is_null and b.ref_cnt == 0 and b.block_id not in free
    }


def is_drained(block_pool: BlockPool) -> bool:
    """True when no request holds a block.

    Deliberately tested on ``ref_cnt`` rather than on the free-queue count:
    permanently-claimed blocks (attention sinks) sit outside the queue with
    ``ref_cnt == 0`` forever, so a free-queue test would report "busy" at
    every barrier on those models and MV-WSA would never fire.
    """
    return all(b.ref_cnt == 0 for b in block_pool.blocks if not b.is_null)


def assert_drained(block_pool: BlockPool) -> None:
    if not is_drained(block_pool):
        held = sum(1 for b in block_pool.blocks if not b.is_null and b.ref_cnt > 0)
        raise RuntimeError(
            f"KV pool resize requires a drained pool; {held} block(s) are "
            "still referenced. Run this only at a busy->idle barrier."
        )


def resize_block_pool(
    block_pool: BlockPool,
    new_num_blocks: int,
    kv_cache_manager: object | None = None,
) -> PoolResizeReport:
    """Grow or shrink ``block_pool`` to ``new_num_blocks``.

    Args:
        block_pool: the live pool, drained.
        new_num_blocks: target size, at least 2.
        kv_cache_manager: the ``KVCacheManager`` owning this pool, when there
            is one. Its ``watermark_blocks`` is derived from the startup pool
            size and is the only admission-headroom term, so a resize that
            leaves it alone turns a 1% reserve into a 15% one after a 15x
            shrink -- which shows up as preemptions, which the controller
            then reads as KV pressure. Pass the manager and it is rescaled.

    Caller contract: on a *grow* the worker has already allocated KV tensors
    big enough to back the new ids; on a *shrink* it frees them only after
    this returns. Getting that order backwards hands the scheduler ids it
    cannot address.

    Raises:
        ValueError: ``new_num_blocks`` below 2, or a shrink that would drop a
            permanently-claimed block.
        RuntimeError: the pool is not drained.
    """
    new_num_blocks = int(new_num_blocks)
    old = int(block_pool.num_gpu_blocks)

    if new_num_blocks < 2:
        # One block is the null block; a pool of only that can seat nothing.
        raise ValueError(f"new_num_blocks must be >= 2, got {new_num_blocks}")

    if new_num_blocks == old:
        return PoolResizeReport(
            old_blocks=old,
            new_blocks=old,
            dropped_cached=0,
            preserved_cached=len(cached_block_ids(block_pool)),
            changed=False,
        )

    assert_drained(block_pool)
    cached_before = cached_block_ids(block_pool)

    if new_num_blocks < old:
        _shrink(block_pool, new_num_blocks)
    else:
        _grow(block_pool, new_num_blocks)

    block_pool.num_gpu_blocks = new_num_blocks
    _rescale_watermark(kv_cache_manager, new_num_blocks)
    _check_invariants(block_pool)

    cached_after = cached_block_ids(block_pool)
    report = PoolResizeReport(
        old_blocks=old,
        new_blocks=new_num_blocks,
        dropped_cached=len(cached_before - cached_after),
        preserved_cached=len(cached_after),
        changed=True,
    )
    logger.debug(
        "kv pool resize: %d -> %d blocks, prefix cache %d kept / %d dropped",
        report.old_blocks,
        report.new_blocks,
        report.preserved_cached,
        report.dropped_cached,
    )
    return report


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------


def _shrink(block_pool: BlockPool, new_num_blocks: int) -> None:
    old = int(block_pool.num_gpu_blocks)
    doomed = set(range(new_num_blocks, old))

    pinned = pinned_block_ids(block_pool) & doomed
    if pinned:
        # Sink blocks and anything else claimed out of band. Relocating them
        # is a bigger change than this module should make on its own, so the
        # controller is told to pick a larger target instead.
        raise ValueError(
            f"cannot shrink to {new_num_blocks}: block(s) {sorted(pinned)[:8]} "
            "are permanently claimed (attention sinks) and would be dropped"
        )

    # Evict only the ids that are going away. Must run before we truncate
    # `blocks`, because evict_blocks indexes into it (and asserts on range).
    # This also drives the metrics collector's per-block cleanup and emits the
    # BlockRemoved events a KV-event subscriber is entitled to see.
    block_pool.evict_blocks(doomed)

    # Unlink each dropped block through the queue's own O(1) remove, which
    # nulls both of its links and keeps num_free_blocks in step. Rebuilding
    # the queue instead would leave these blocks pointing into a chain that
    # ends at an orphaned sentinel -- see the module docstring.
    queue = block_pool.free_block_queue
    free = _free_block_ids(block_pool)
    for block_id in doomed:
        if block_id in free:
            queue.remove(block_pool.blocks[block_id])

    # Same objects for the survivors, so every surviving prefix-cache entry
    # keeps pointing at a block that is still in the pool.
    del block_pool.blocks[new_num_blocks:]

    # Drop per-block hash bookkeeping for ids that no longer exist.
    # evict_blocks clears entries for cached blocks; an uncached doomed id may
    # still hold a stale set here.
    for block_id in doomed:
        block_pool.cached_block_hashes_by_block.pop(block_id, None)


def _grow(block_pool: BlockPool, new_num_blocks: int) -> None:
    old = int(block_pool.num_gpu_blocks)
    fresh = [KVCacheBlock(block_id) for block_id in range(old, new_num_blocks)]
    block_pool.blocks.extend(fresh)

    # Fresh blocks cache nothing, so they are the cheapest thing to hand out:
    # putting them at the head means the next allocations consume them instead
    # of evicting a warm prefix-cache block. This mirrors what `free_blocks`
    # already does with hash-less blocks.
    block_pool.free_block_queue.prepend_n(fresh)


def _rescale_watermark(kv_cache_manager: object | None, new_num_blocks: int) -> None:
    """Re-derive the admission headroom for the new pool size."""
    if kv_cache_manager is None:
        return
    watermark = getattr(kv_cache_manager, "watermark", None)
    if watermark is None or not hasattr(kv_cache_manager, "watermark_blocks"):
        logger.warning(
            "kv pool resize: %s exposes no watermark to rescale; admission "
            "headroom stays sized for the old pool",
            type(kv_cache_manager).__name__,
        )
        return
    kv_cache_manager.watermark_blocks = int(watermark * new_num_blocks)


# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------


def _check_invariants(block_pool: BlockPool) -> None:
    """Cheap structural checks. These are bugs, not runtime conditions."""
    n = block_pool.num_gpu_blocks
    assert len(block_pool.blocks) == n, (
        f"blocks list ({len(block_pool.blocks)}) out of sync with "
        f"num_gpu_blocks ({n})"
    )
    assert block_pool.blocks[0] is block_pool.null_block, (
        "null_block identity lost; single-type managers cached the old object"
    )
    walked = block_pool.free_block_queue.get_all_free_blocks()
    assert len(walked) == block_pool.free_block_queue.num_free_blocks, (
        f"free list walks {len(walked)} blocks but reports "
        f"{block_pool.free_block_queue.num_free_blocks}"
    )
    assert all(b.block_id < n for b in walked), (
        "free list still holds a block beyond the pool"
    )
    # Every non-null block is either free or permanently claimed; a drained
    # pool has nothing referenced.
    expected_free = n - 1 - len(pinned_block_ids(block_pool))
    assert len(walked) == expected_free, (
        f"free queue holds {len(walked)}, expected {expected_free} for a "
        f"drained pool of {n}"
    )
    # cached_block_hash_to_block has no public iteration, but it is kept in
    # step with cached_block_hashes_by_block, which is keyed by block id --
    # so a stale id here is the same defect, visible through a public field.
    stale = [bid for bid in block_pool.cached_block_hashes_by_block if bid >= n]
    assert not stale, (
        f"prefix-cache bookkeeping still references block ids {stale[:8]} "
        f"beyond the pool ({n})"
    )
