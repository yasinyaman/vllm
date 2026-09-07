# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-only checks for the live GPU slot-tier resize.

test_expert_cache_resize.py covers the bookkeeping on any device. The point of
MV-WSA is that a shrink actually hands device bytes back so the KV pool can
claim them, and that is only observable on a real GPU -- so it lives here.
"""

import pytest
import torch

from vllm.model_executor.layers.fused_moe.expert_weight_provider import (
    CachedWeightProvider,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

E = 8
H = 256
INTER = 128


def make_provider(
    capacity: int, max_capacity: int | None = None, with_scales: bool = False
) -> CachedWeightProvider:
    dev = torch.device("cuda")
    kw = {}
    if with_scales:
        kw = dict(
            w13_scale=torch.randn(E, 2 * INTER, 1, device=dev),
            w2_scale=torch.randn(E, H, 1, device=dev),
        )
    return CachedWeightProvider(
        capacity=capacity,
        w13_weight=torch.randn(E, 2 * INTER, H, device=dev, dtype=torch.bfloat16),
        w2_weight=torch.randn(E, H, INTER, device=dev, dtype=torch.bfloat16),
        split="expert",
        max_capacity=max_capacity,
        **kw,
    )


def test_buffers_live_on_the_device():
    p = make_provider(4, max_capacity=8)
    assert p.buf_w13.is_cuda and p.buf_w2.is_cuda
    assert p.slot_bytes == p.buf_w13[0].nbytes + p.buf_w2[0].nbytes


def test_shrink_actually_returns_device_bytes():
    """The whole point: freed slots become KV blocks, not allocator cache."""
    p = make_provider(8, max_capacity=8)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    p.resize(2)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert before - after == 6 * p.slot_bytes, (
        f"expected {6 * p.slot_bytes} bytes back, got {before - after}"
    )


def test_grow_consumes_exactly_the_expected_bytes():
    p = make_provider(2, max_capacity=8)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    p.resize(6)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert after - before == 4 * p.slot_bytes


def test_grow_copies_live_slots_on_device():
    p = make_provider(2, max_capacity=8)
    p.buf_w13[0].fill_(2.0)
    p.buf_w2[0].fill_(-3.0)
    p.resize(8)
    torch.cuda.synchronize()
    assert torch.all(p.buf_w13[0] == 2.0)
    assert torch.all(p.buf_w2[0] == -3.0)


def test_resize_waits_for_an_in_flight_host_to_device_copy():
    """resize() synchronizes before dropping the old buffers."""
    p = make_provider(4, max_capacity=8)
    src = torch.full(
        (2 * INTER, H), 5.0, dtype=torch.bfloat16, device="cpu"
    ).pin_memory()
    p.buf_w13[1].copy_(src, non_blocking=True)
    p.resize(8)  # must not race the copy above
    torch.cuda.synchronize()
    assert torch.all(p.buf_w13[1] == 5.0)


def test_scale_buffer_identity_and_device_survive_resize():
    p = make_provider(2, max_capacity=8, with_scales=True)
    w13_scale = p.buf_w13_scale
    assert w13_scale.is_cuda and w13_scale.shape[0] == 8
    p.resize(8)
    p.resize(3)
    assert p.buf_w13_scale is w13_scale
    assert p.buf_w13_scale.shape[0] == 8


def test_scale_buffer_bytes_do_not_move_with_capacity():
    """Pinned at the ceiling, so a resize trades weight bytes only."""
    p = make_provider(8, max_capacity=8, with_scales=True)
    scale_bytes = p.buf_w13_scale.nbytes + p.buf_w2_scale.nbytes
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    p.resize(1)
    torch.cuda.synchronize()
    freed = before - torch.cuda.memory_allocated()
    assert freed == 7 * p.slot_bytes
    assert freed < scale_bytes + 7 * p.slot_bytes  # scales stayed put
