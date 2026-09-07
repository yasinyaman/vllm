# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MV-WSA controller under a fake engine with a real block pool.

No GPU. What this file locks: when a cycle runs at all (every guard yields a
hold with no RPC), the order of the two actuators as a function of which side
grows, the three num_blocks copies staying in sync, the prefix-cache reset
after a rebuild, the reconciliation after a failed worker half, and the budget
staying constant across moves.
"""

import json
from collections import deque
from types import SimpleNamespace

import pytest

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.engine.mvwsa_controller import MVWSAController

BLOCK = 16
KV_BLOCK_BYTES = 1 << 20  # one logical block, summed over layers
SLOT_BYTES = 4 << 20  # one expert slot, summed over layers


class FakeScheduler:
    def __init__(self, pool: BlockPool):
        self.kv_cache_manager = SimpleNamespace(
            block_pool=pool, watermark=0.01, watermark_blocks=1
        )
        self.kv_cache_config = SimpleNamespace(
            num_blocks=pool.num_gpu_blocks, kv_cache_groups=[object()]
        )
        self.deferred_frees = deque()
        self.busy = False
        self.num_preemptions = 0
        self.resets = 0

    def has_requests(self):
        return self.busy

    def has_unfinished_requests(self):
        return self.busy

    def reset_prefix_cache(self, **kwargs):
        self.resets += 1
        return True


class FakeEngine:
    """Records every RPC and plays the worker's part of a move."""

    def __init__(self, kv_blocks=64, cap=8, *, cap_min=2, cap_max=32, resizable=True):
        pool = BlockPool(
            num_gpu_blocks=kv_blocks, enable_caching=True, hash_block_size=BLOCK
        )
        self.scheduler = FakeScheduler(pool)
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=BLOCK, num_gpu_blocks=kv_blocks),
            model_config=SimpleNamespace(max_model_len=64),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
        )
        self.kv, self.cap = kv_blocks, cap
        self.calls: list[tuple] = []
        self.union = 0
        self.fail_next = False
        self.geometry = dict(
            layers=2,
            resizable=resizable,
            kv_block_bytes=KV_BLOCK_BYTES,
            cap_min=cap_min,
            cap_max=cap_max,
            expert_slot_bytes=SLOT_BYTES,
        )

    @property
    def pool(self):
        return self.scheduler.kv_cache_manager.block_pool

    def collective_rpc(self, method, args=(), **kwargs):
        self.calls.append((method, tuple(args)))
        if method == "mvwsa_geometry":
            return [dict(self.geometry, kv_blocks=self.kv, capacity=self.cap)]
        if method == "mvwsa_take_union_peak":
            union, self.union = self.union, 0
            return [union]
        if method == "mvwsa_apply":
            kv_to, cap_to, grow = args
            base = dict(
                experts_grow=grow,
                cap_from=self.cap,
                cap_to=cap_to,
                kv_from=self.kv,
                kv_to=kv_to,
                kv=None,
                failed=None,
            )
            if self.fail_next:
                self.fail_next = False
                return [
                    dict(
                        base,
                        failed="RuntimeError: OOM",
                        cap_in_force=self.cap,
                        kv_in_force=self.kv,
                    )
                ]
            if kv_to != self.kv:
                base["kv"] = dict(
                    kv_blocks=kv_to, old_kv_blocks=self.kv, content_preserved=False
                )
            self.kv, self.cap = kv_to, cap_to
            return [dict(base, cap_in_force=cap_to, kv_in_force=kv_to)]
        raise AssertionError(method)


def epoch(ctrl, engine, *, steps=8, union=0):
    """One busy period of ``steps`` engine steps, then the idle barrier."""
    engine.union = union
    engine.scheduler.busy = True
    for _ in range(steps):
        ctrl.observe_step()
    engine.scheduler.busy = False
    return ctrl.maybe_rebalance()


def methods(engine):
    return [m for m, _ in engine.calls]


# ----------------------------------------------------------------------
# guards: no RPC unless the engine is drained
# ----------------------------------------------------------------------


def test_nothing_happens_before_the_first_step():
    engine = FakeEngine()
    ctrl = MVWSAController(engine, "expert-union")
    assert ctrl.maybe_rebalance() is None
    assert engine.calls == []


def test_busy_engine_is_a_hold_without_rpc():
    engine = FakeEngine()
    ctrl = MVWSAController(engine, "expert-union")
    ctrl.observe_step()
    engine.scheduler.busy = True
    assert ctrl.maybe_rebalance() is None
    assert engine.calls == []


def test_deferred_frees_and_held_blocks_are_holds():
    engine = FakeEngine()
    ctrl = MVWSAController(engine, "expert-union")
    ctrl.observe_step()
    engine.scheduler.deferred_frees.append((1, []))
    assert ctrl.maybe_rebalance() is None
    engine.scheduler.deferred_frees.clear()
    held = engine.pool.get_new_blocks(2)
    assert ctrl.maybe_rebalance() is None
    engine.pool.free_blocks(held)
    assert engine.calls == []


def test_non_resizable_geometry_disables_for_good():
    engine = FakeEngine(resizable=False)
    ctrl = MVWSAController(engine, "expert-union")
    assert epoch(ctrl, engine, union=30) is None
    assert ctrl.disabled and "resize" in ctrl.disabled
    assert methods(engine) == ["mvwsa_geometry"]
    assert epoch(ctrl, engine, union=30) is None
    assert methods(engine) == ["mvwsa_geometry"]  # never asked again


def test_pinned_blocks_disable_for_good():
    engine = FakeEngine()
    engine.pool.free_block_queue.popleft()  # a sink block: out of the queue, ref 0
    ctrl = MVWSAController(engine, "expert-union")
    assert epoch(ctrl, engine, union=30) is None
    assert ctrl.disabled and "pinned" in ctrl.disabled


# ----------------------------------------------------------------------
# a full cycle: experts grow, KV shrinks -- scheduler first
# ----------------------------------------------------------------------


def test_experts_grow_resizes_the_pool_before_the_worker():
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union")
    first = epoch(ctrl, engine, union=16)
    assert first is not None and not first.applied  # confirmation epoch 1/2
    assert engine.pool.num_gpu_blocks == 64

    d = epoch(ctrl, engine, union=16)
    assert d is not None and d.applied and d.experts_grow
    # budget 64 MiB + 32 MiB = 96 MiB; cap 16 costs 64 MiB, leaving 32 blocks
    assert (d.kv_to, d.cap_to) == (32, 16)
    assert engine.calls[-1] == ("mvwsa_apply", (32, 16, True))
    assert engine.pool.num_gpu_blocks == 32
    assert engine.scheduler.kv_cache_config.num_blocks == 32
    assert engine.vllm_config.cache_config.num_gpu_blocks == 32
    assert engine.scheduler.resets == 1  # tensors came back zeroed
    assert ctrl.cap_now == 16 and ctrl.moves == 1


def test_kv_grow_moves_the_worker_before_the_pool():
    engine = FakeEngine(kv_blocks=32, cap=16)
    ctrl = MVWSAController(engine, "expert-union")
    epoch(ctrl, engine, union=4)
    d = epoch(ctrl, engine, union=4)
    assert d is not None and d.applied and not d.experts_grow
    # 32 MiB + 64 MiB = 96 MiB; cap 4 costs 16 MiB, leaving 80 blocks
    assert (d.kv_to, d.cap_to) == (80, 4)
    assert engine.calls[-1] == ("mvwsa_apply", (80, 4, False))
    assert engine.pool.num_gpu_blocks == 80
    assert engine.scheduler.kv_cache_config.num_blocks == 80
    assert engine.scheduler.resets == 1


def test_budget_is_constant_across_moves():
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union")
    epoch(ctrl, engine, union=16)
    epoch(ctrl, engine, union=16)
    budget = ctrl.policy.geometry.total_budget_bytes
    epoch(ctrl, engine, union=4)
    d = epoch(ctrl, engine, union=4)
    assert d is not None and d.applied
    assert ctrl.policy.geometry.total_budget_bytes == budget
    assert ctrl.policy.geometry.cost(d.kv_to, d.cap_to) <= budget


# ----------------------------------------------------------------------
# a failed worker half: the pool follows what the worker reports in force
# ----------------------------------------------------------------------


def test_failed_kv_shrink_grows_the_pool_back_and_keeps_the_cache():
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union")
    epoch(ctrl, engine, union=16)
    engine.fail_next = True
    d = epoch(ctrl, engine, union=16)
    assert d is not None and d.applied
    assert engine.pool.num_gpu_blocks == 64  # shrunk to 32, then reconciled
    assert engine.scheduler.kv_cache_config.num_blocks == 64
    assert ctrl.cap_now == 8 and ctrl.moves == 0
    assert engine.scheduler.resets == 0  # no rebuild happened


# ----------------------------------------------------------------------
# what the policy is told
# ----------------------------------------------------------------------


def test_kv_demand_is_the_live_peak_plus_the_prefix_cache():
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union")
    epoch(ctrl, engine)  # builds the policy
    seen = []
    real = ctrl.policy.decide
    ctrl.policy.decide = lambda obs: (seen.append(obs), real(obs))[1]

    engine.scheduler.busy = True
    held = engine.pool.get_new_blocks(5)
    for _ in range(8):
        ctrl.observe_step()
    engine.pool.free_blocks(held)
    for bid in (3, 5):
        engine.pool._insert_block_hash(
            make_block_hash_with_group_id(BlockHash(f"h{bid}".encode()), 0),
            engine.pool.blocks[bid],
            num_tokens=BLOCK,
        )
    engine.scheduler.busy = False
    engine.union = 6
    ctrl.maybe_rebalance()
    obs = seen[-1]
    assert obs.kv_demand_blocks == 5 + 2
    assert obs.expert_union_peak == 6
    assert obs.steps == 8
    assert obs.kv_pressure is False


def test_kv_pressure_is_a_preemption_delta_only():
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union")
    epoch(ctrl, engine)
    seen = []
    real = ctrl.policy.decide
    ctrl.policy.decide = lambda obs: (seen.append(obs), real(obs))[1]
    engine.scheduler.num_preemptions = 3
    epoch(ctrl, engine)
    assert seen[-1].kv_pressure is True
    epoch(ctrl, engine)  # no new preemptions since
    assert seen[-1].kv_pressure is False


def test_force_runs_a_cycle_without_an_open_epoch():
    """The in-process client never turns idle; a harness forces the barrier."""
    engine = FakeEngine()
    ctrl = MVWSAController(engine, "expert-union")
    assert ctrl.maybe_rebalance() is None
    assert ctrl.maybe_rebalance(force=True) is not None
    assert methods(engine)[:2] == ["mvwsa_geometry", "mvwsa_take_union_peak"]


def test_decisions_are_logged_as_jsonl(tmp_path):
    path = tmp_path / "mvwsa.jsonl"
    engine = FakeEngine(kv_blocks=64, cap=8)
    ctrl = MVWSAController(engine, "expert-union", str(path))
    epoch(ctrl, engine, union=16)
    epoch(ctrl, engine, union=16)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["decision"]["applied"] is False
    assert lines[1]["decision"]["applied"] is True
    assert lines[1]["report"]["kv_in_force"] == 32
    assert lines[1]["obs"]["expert_union_peak"] == 16


def test_kv_peak_strategy_is_accepted():
    engine = FakeEngine()
    ctrl = MVWSAController(engine, "kv-peak")
    assert epoch(ctrl, engine) is not None
    with pytest.raises(ValueError):
        MVWSAController(engine, "nonsense")._init_policy()
