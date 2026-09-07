# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The KV teardown/rebuild primitive on a real engine, in process.

What the fakes cannot prove: that a rebuilt cache serves the same tokens as
before once the prefix cache is reset (and would not without it -- the NaN
poison), that the bytes move by exactly the block count times the pool's
per-block price, that the worker RPC path resizes a real model's providers
alongside, and that the controller's forced barrier drives the whole thing.

Needs a CUDA device and the tiny MoE checkpoint. Runs the engine in process
(VLLM_ENABLE_V1_MULTIPROCESSING=0) so the scheduler and the model runner are
reachable, and without the moe-surgeon plugin if it is installed.
"""

import hashlib
import json
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

MODEL = os.environ.get("MVWSA_TEST_MODEL", "nm-testing/tinysmokeqwen3moe")
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain in one sentence why the sky appears blue:",
]


@pytest.fixture(scope="module")
def engine():
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.setdefault("VLLM_PLUGINS", "lora_filesystem_resolver")
    os.environ["VLLM_MOE_MVWSA"] = "expert-union"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        moe_expert_cache_size=4,
        moe_expert_cache_max_size=8,
        enable_prefix_caching=True,
        enforce_eager=False,  # piecewise graphs must survive the rebuild
        gpu_memory_utilization=0.3,
        max_model_len=512,
        max_num_seqs=4,
        disable_log_stats=True,
    )
    core = llm.llm_engine.engine_core.engine_core
    runner = core.model_executor.driver_worker.worker.model_runner
    params = SamplingParams(temperature=0.0, max_tokens=48)

    def gen():
        outs = llm.generate(PROMPTS, params)
        ids = [list(o.outputs[0].token_ids) for o in outs]
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()

    yield llm, core, runner, gen


def _pool(core):
    return core.scheduler.kv_cache_manager.block_pool


def _sync(core, n):
    core.vllm_config.cache_config.num_gpu_blocks = n
    core.scheduler.kv_cache_config.num_blocks = n


def _poison(runner):
    """Fill the fresh tensors with NaN: a stale prefix-cache hit would show."""
    for t in runner.kv_caches:
        if t.is_floating_point():
            t.fill_(float("nan"))


def test_shrink_then_grow_serves_the_same_tokens(engine):
    from vllm.v1.core.kv_cache_utils import _pool_bytes_per_block
    from vllm.v1.core.kv_pool_resize import resize_block_pool

    llm, core, runner, gen = engine
    pool = _pool(core)
    manager = core.scheduler.kv_cache_manager
    n0 = pool.num_gpu_blocks
    per_block = _pool_bytes_per_block(
        core.vllm_config, runner.kv_cache_config.kv_cache_groups
    )
    ref = gen()  # also fills the prefix cache with these prompts

    # --- shrink: scheduler first, then the worker ---
    n_down = max(n0 // 2, 8)
    resize_block_pool(pool, n_down, manager)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    report = runner.resize_kv_cache(n_down)
    torch.cuda.synchronize()
    freed = before - torch.cuda.memory_allocated()
    assert report["kv_blocks"] == n_down and report["content_preserved"] is False
    assert runner.kv_cache_config.num_blocks == n_down
    assert abs(freed - (n0 - n_down) * per_block) <= per_block, (freed, per_block)
    _sync(core, n_down)
    _poison(runner)
    assert core.scheduler.reset_prefix_cache()
    assert gen() == ref

    # --- grow: worker first, then the scheduler ---
    before = torch.cuda.memory_allocated()
    report = runner.resize_kv_cache(n0)
    torch.cuda.synchronize()
    taken = torch.cuda.memory_allocated() - before
    assert abs(taken - (n0 - n_down) * per_block) <= per_block
    resize_block_pool(pool, n0, manager)
    _sync(core, n0)
    _poison(runner)
    assert core.scheduler.reset_prefix_cache()
    assert gen() == ref


def test_worker_rpc_moves_both_pools(engine):
    from vllm.v1.core.kv_pool_resize import resize_block_pool

    llm, core, runner, gen = engine
    pool = _pool(core)
    manager = core.scheduler.kv_cache_manager
    ref = gen()
    n0 = pool.num_gpu_blocks
    g = core.collective_rpc("mvwsa_geometry")[0]
    assert g["resizable"] and g["capacity"] == 4 and g["cap_max"] == 8
    assert g["kv_blocks"] == n0 and g["layers"] > 0

    # experts grow 4 -> 8, KV shrinks: scheduler first, then the worker
    n_down = max(n0 // 2, 8)
    resize_block_pool(pool, n_down, manager)
    r = core.collective_rpc("mvwsa_apply", args=(n_down, 8, True))[0]
    assert r["failed"] is None and (r["kv_in_force"], r["cap_in_force"]) == (n_down, 8)
    _sync(core, n_down)
    assert core.scheduler.reset_prefix_cache()
    assert gen() == ref

    # back: experts shrink 8 -> 4, KV grows: worker first, then the scheduler
    r = core.collective_rpc("mvwsa_apply", args=(n0, 4, False))[0]
    assert r["failed"] is None and (r["kv_in_force"], r["cap_in_force"]) == (n0, 4)
    resize_block_pool(pool, n0, manager)
    _sync(core, n0)
    assert core.scheduler.reset_prefix_cache()
    assert gen() == ref
    assert core.collective_rpc("mvwsa_geometry")[0]["capacity"] == 4


def test_controller_forced_barrier_moves_the_split(engine):
    llm, core, runner, gen = engine
    ctrl = core.mvwsa
    assert ctrl is not None, "VLLM_MOE_MVWSA was set before the engine was built"
    pool = _pool(core)
    ref = gen()
    n0, cap0 = pool.num_gpu_blocks, ctrl.cap_now or 4
    # The in-process client never turns idle: simulate epochs by hand. Under
    # expert-union with bs<=3 the union is tiny, so the law wants the floor
    # (top_k under the token split) and hands the rest to KV.
    decisions = []
    for _ in range(3):
        gen()
        for _ in range(8):
            ctrl.observe_step()
        decisions.append(ctrl.maybe_rebalance(force=True))
    assert ctrl.disabled is None, ctrl.disabled
    applied = [d for d in decisions if d is not None and d.applied]
    assert applied, [d.reason for d in decisions if d is not None]
    d = applied[-1]
    assert d.cap_to < cap0 and d.kv_to > n0
    assert pool.num_gpu_blocks == d.kv_to
    assert core.scheduler.kv_cache_config.num_blocks == d.kv_to
    assert core.collective_rpc("mvwsa_geometry")[0]["capacity"] == d.cap_to
    assert gen() == ref
