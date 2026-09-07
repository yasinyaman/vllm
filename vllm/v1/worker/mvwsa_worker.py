# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker half of an MV-WSA move: find the providers, measure, apply.

The engine decides (``vllm.v1.core.mvwsa_policy``) and resizes the scheduler's
block pool (``vllm.v1.core.kv_pool_resize``); this module moves the bytes on
the device -- every MoE layer's ``CachedWeightProvider.resize`` and the model
runner's ``resize_kv_cache``. Pure functions over duck-typed objects, so the
ordering and rollback rules run under fakes; ``Worker.mvwsa_*`` in
gpu_worker.py are one-line wrappers reached by name through ``collective_rpc``.

Ordering is the whole point. Peak device memory during a move is
``max(old, new)`` per pool, never ``old + new``, only if the side giving bytes
up moves first: KV shrinks before the experts grow, experts shrink before KV
grows. ``experts_grow`` carries that decision from the policy.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def iter_providers(compilation_config: Any) -> list[Any]:
    """Every MoE layer's expert-cache provider, in layer order.

    The forward context is the registry: ``register_layer_for_moe_forward_op``
    files each ``MoERunner`` under its layer name and appends the name to
    ``static_all_moe_layers``. Layers without a cache have no provider and
    are skipped.
    """
    ctx = compilation_config.static_forward_context
    providers = []
    for name in compilation_config.static_all_moe_layers:
        routed = getattr(ctx.get(name), "routed_experts", None)
        provider = getattr(routed, "expert_weight_provider", None)
        if provider is not None:
            providers.append(provider)
    return providers


def geometry(model_runner: Any, vllm_config: Any) -> dict[str, Any]:
    """What the policy needs once: prices, bounds, and the split now in force."""
    from vllm.v1.core.kv_cache_utils import _pool_bytes_per_block

    providers = iter_providers(vllm_config.compilation_config)
    kv_config = model_runner.kv_cache_config
    return {
        "layers": len(providers),
        # Capacity moves in lockstep across layers, so one layer that cannot
        # follow a moving buffer pins every other.
        "resizable": bool(providers) and all(p.resizable for p in providers),
        "kv_blocks": kv_config.num_blocks,
        "kv_block_bytes": _pool_bytes_per_block(vllm_config, kv_config.kv_cache_groups),
        "capacity": min((p.capacity for p in providers), default=0),
        "cap_min": max((p.min_capacity for p in providers), default=0),
        "cap_max": min((p.max_capacity for p in providers), default=0),
        "expert_slot_bytes": sum(p.slot_bytes for p in providers),
    }


def take_union_peak(providers: list[Any]) -> int:
    """The widest per-call expert union any layer saw this epoch; resets all."""
    return max((p.take_union_peak() for p in providers), default=0)


def apply(
    model_runner: Any,
    vllm_config: Any,
    kv_blocks: int,
    capacity: int,
    experts_grow: bool,
) -> dict[str, Any]:
    """Move both pools to ``(kv_blocks, capacity)`` in the memory-safe order.

    On any failure the device is put back at the split the scheduler still
    holds -- the experts first, since shrinking them frees the bytes a KV
    rebuild needs -- and the report carries ``failed`` plus the sizes in
    force, which is what the engine reconciles its pool against. A rollback
    that fails too propagates: at that point the process has no coherent KV
    cache and pretending otherwise would serve wrong tokens.
    """
    providers = iter_providers(vllm_config.compilation_config)
    kv_from = model_runner.kv_cache_config.num_blocks
    cap_from = providers[0].capacity if providers else capacity
    report: dict[str, Any] = {
        "experts_grow": experts_grow,
        "cap_from": cap_from,
        "cap_to": capacity,
        "kv_from": kv_from,
        "kv_to": kv_blocks,
        "kv": None,
        "failed": None,
    }

    def move_experts() -> None:
        if capacity != cap_from:
            for provider in providers:
                provider.resize(capacity)

    def move_kv() -> None:
        if kv_blocks != kv_from:
            report["kv"] = model_runner.resize_kv_cache(kv_blocks)

    try:
        if experts_grow:
            move_kv()
            move_experts()
        else:
            move_experts()
            move_kv()
    except Exception as exc:
        report["failed"] = f"{type(exc).__name__}: {exc}"
        logger.error("MV-WSA move failed (%s); rolling back", report["failed"])
        for provider in providers:
            if provider.capacity != cap_from:
                provider.resize(cap_from)
        in_force = getattr(model_runner, "kv_cache_config", None)
        if in_force is None or in_force.num_blocks != kv_from:
            report["kv"] = model_runner.resize_kv_cache(kv_from)

    report["cap_in_force"] = providers[0].capacity if providers else capacity
    report["kv_in_force"] = model_runner.kv_cache_config.num_blocks
    return report
