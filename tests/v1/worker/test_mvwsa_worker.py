# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The worker half of an MV-WSA move, under fakes: ordering and rollback.

No GPU, no model. The real primitives (CachedWeightProvider.resize and
GPUModelRunner.resize_kv_cache) are covered by their own CUDA tests; what this
file locks is the contract between them -- which side moves first, what a
failed move leaves in force, and what the report tells the engine.
"""

from types import SimpleNamespace

import pytest

from vllm.v1.worker import mvwsa_worker as mw


class FakeProvider:
    def __init__(
        self,
        capacity,
        *,
        min_capacity=1,
        max_capacity=8,
        slot_bytes=100,
        resizable=True,
        fail_at=None,
        log=None,
        name="p",
    ):
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.slot_bytes = slot_bytes
        self.resizable = resizable
        self.union_peak = 0
        self._fail_at = fail_at
        self._log = log if log is not None else []
        self._name = name

    def resize(self, new_capacity):
        if self._fail_at is not None and new_capacity == self._fail_at:
            raise RuntimeError(f"{self._name}: no bytes for {new_capacity}")
        self._log.append(("exp", self._name, new_capacity))
        self.capacity = new_capacity

    def take_union_peak(self):
        peak, self.union_peak = self.union_peak, 0
        return peak


class FakeRunner:
    def __init__(self, kv_blocks, *, fail_at=(), log=None):
        self.kv_cache_config = SimpleNamespace(num_blocks=kv_blocks, kv_cache_groups=[])
        self._fail_at = set(fail_at)
        self._log = log if log is not None else []

    def resize_kv_cache(self, n):
        # Teardown happens before the new allocation: a failure leaves no
        # cache, and a rebuild after one starts from nothing (the real
        # primitive reads the old config with a default for the same reason).
        old_config = getattr(self, "kv_cache_config", None)
        old = old_config.num_blocks if old_config is not None else None
        if old_config is not None:
            del self.kv_cache_config
        if n in self._fail_at:
            raise RuntimeError(f"OOM at {n} blocks")
        self.kv_cache_config = SimpleNamespace(num_blocks=n, kv_cache_groups=[])
        self._log.append(("kv", n))
        return {"kv_blocks": n, "old_kv_blocks": old, "content_preserved": False}


def make_config(providers, extra_layers=()):
    ctx = {}
    names = []
    for i, p in enumerate(providers):
        name = f"moe.{i}"
        ctx[name] = SimpleNamespace(
            routed_experts=SimpleNamespace(expert_weight_provider=p)
        )
        names.append(name)
    for name in extra_layers:  # MoE layers registered without a cache
        ctx[name] = SimpleNamespace(
            routed_experts=SimpleNamespace(expert_weight_provider=None)
        )
        names.append(name)
    cc = SimpleNamespace(static_forward_context=ctx, static_all_moe_layers=names)
    return SimpleNamespace(compilation_config=cc)


def test_iter_providers_keeps_layer_order_and_skips_uncached_layers():
    log = []
    ps = [FakeProvider(4, log=log, name=f"p{i}") for i in range(3)]
    cfg = make_config(ps, extra_layers=["dense-moe"])
    assert mw.iter_providers(cfg.compilation_config) == ps


def test_geometry_aggregates_the_conservative_bounds(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils._pool_bytes_per_block", lambda cfg, groups: 4096
    )
    ps = [
        FakeProvider(4, min_capacity=1, max_capacity=16, slot_bytes=100),
        FakeProvider(4, min_capacity=8, max_capacity=12, slot_bytes=150),
    ]
    g = mw.geometry(FakeRunner(500), make_config(ps))
    assert g == {
        "layers": 2,
        "resizable": True,
        "kv_blocks": 500,
        "kv_block_bytes": 4096,
        "capacity": 4,
        "cap_min": 8,  # the strictest floor wins
        "cap_max": 12,  # the lowest ceiling wins
        "expert_slot_bytes": 250,
    }


def test_geometry_is_not_resizable_if_any_layer_is_not(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.core.kv_cache_utils._pool_bytes_per_block", lambda cfg, groups: 1
    )
    ps = [FakeProvider(4), FakeProvider(4, resizable=False)]
    assert mw.geometry(FakeRunner(10), make_config(ps))["resizable"] is False
    assert mw.geometry(FakeRunner(10), make_config([]))["resizable"] is False


def test_take_union_peak_is_the_max_over_layers_and_resets_every_layer():
    ps = [FakeProvider(4), FakeProvider(4), FakeProvider(4)]
    ps[0].union_peak, ps[1].union_peak, ps[2].union_peak = 3, 7, 5
    assert mw.take_union_peak(ps) == 7
    assert [p.union_peak for p in ps] == [0, 0, 0]
    assert mw.take_union_peak([]) == 0


def test_experts_grow_shrinks_kv_first():
    log = []
    ps = [FakeProvider(4, log=log, name="a"), FakeProvider(4, log=log, name="b")]
    runner = FakeRunner(500, log=log)
    r = mw.apply(runner, make_config(ps), kv_blocks=300, capacity=8, experts_grow=True)
    assert log == [("kv", 300), ("exp", "a", 8), ("exp", "b", 8)]
    assert r["failed"] is None
    assert (r["kv_in_force"], r["cap_in_force"]) == (300, 8)
    assert r["kv"]["content_preserved"] is False


def test_kv_grow_shrinks_experts_first():
    log = []
    ps = [FakeProvider(8, log=log, name="a"), FakeProvider(8, log=log, name="b")]
    runner = FakeRunner(300, log=log)
    r = mw.apply(runner, make_config(ps), kv_blocks=500, capacity=4, experts_grow=False)
    assert log == [("exp", "a", 4), ("exp", "b", 4), ("kv", 500)]
    assert (r["kv_in_force"], r["cap_in_force"]) == (500, 4)


def test_unchanged_sides_are_not_touched():
    log = []
    ps = [FakeProvider(4, log=log)]
    runner = FakeRunner(500, log=log)
    r = mw.apply(runner, make_config(ps), kv_blocks=500, capacity=4, experts_grow=True)
    assert log == []
    assert r["kv"] is None and r["failed"] is None
    r = mw.apply(runner, make_config(ps), kv_blocks=400, capacity=4, experts_grow=True)
    assert log == [("kv", 400)]


def test_a_failed_expert_grow_rolls_every_layer_and_kv_back():
    log = []
    ps = [
        FakeProvider(4, log=log, name="a"),
        FakeProvider(4, log=log, name="b", fail_at=8),
        FakeProvider(4, log=log, name="c"),
    ]
    runner = FakeRunner(500, log=log)
    r = mw.apply(runner, make_config(ps), kv_blocks=300, capacity=8, experts_grow=True)
    assert r["failed"] is not None and "b: no bytes" in r["failed"]
    assert [p.capacity for p in ps] == [4, 4, 4]
    assert runner.kv_cache_config.num_blocks == 500
    assert (r["kv_in_force"], r["cap_in_force"]) == (500, 4)
    # KV shrank, layer a grew, b failed, a rolled back, KV rebuilt at the old size.
    assert log == [("kv", 300), ("exp", "a", 8), ("exp", "a", 4), ("kv", 500)]


def test_a_failed_kv_rebuild_restores_the_old_pool_and_the_experts():
    log = []
    ps = [FakeProvider(8, log=log, name="a")]
    runner = FakeRunner(300, fail_at={500}, log=log)
    r = mw.apply(runner, make_config(ps), kv_blocks=500, capacity=4, experts_grow=False)
    assert r["failed"] is not None and "OOM at 500" in r["failed"]
    assert runner.kv_cache_config.num_blocks == 300
    assert ps[0].capacity == 8
    assert (r["kv_in_force"], r["cap_in_force"]) == (300, 8)
    assert log == [("exp", "a", 4), ("exp", "a", 8), ("kv", 300)]


def test_a_rollback_that_fails_propagates():
    """No coherent KV cache left: say so loudly rather than serve from nothing."""
    log = []
    ps = [FakeProvider(8, log=log, name="a")]
    runner = FakeRunner(300, fail_at={500, 300}, log=log)
    with pytest.raises(RuntimeError, match="OOM at 300"):
        mw.apply(runner, make_config(ps), kv_blocks=500, capacity=4, experts_grow=False)
    # The experts were put back before the KV rebuild was attempted.
    assert ps[0].capacity == 8
