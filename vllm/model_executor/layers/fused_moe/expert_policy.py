# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction scoring policies for the expert weight cache.

The cache tiers in ``CachedWeightProvider`` own slots and residency; a policy
owns per-expert statistics and the eviction score. The provider's victim scans
iterate their own residency dicts and ask the policy to score each candidate,
so scan order -- and therefore tie-breaking -- is the provider's, unchanged.

Two policies:

* ``LFRUPolicy`` -- the shipped behavior, ``score = freq / age`` computed from
  the residency entry's own counters. Stateless; bit-identical to the inline
  expression it replaced.
* ``EWMAPolicy`` -- ``score = w_f * ewma + w_r * recency + w_m * prior``, with
  statistics kept outside the residency entry so they survive eviction.
  Offline replay of routing traces (bench/hit_rate_sweep.py) measured the
  additive form at 7-29% fewer disk reads than LFRU depending on capacity;
  the ratio form with persistent counters is *worse* than LFRU, which is why
  persistence is not offered as a knob on the ratio.

Each cache tier holds its own policy instance: the GPU and RAM tiers already
run separate clocks and must keep separate statistics.
"""

import math

import vllm.envs as envs


def make_policy(num_experts: int) -> "LFRUPolicy":
    """Build the configured policy for one cache tier."""
    if envs.VLLM_MOE_CACHE_POLICY == "ewma":
        return EWMAPolicy(num_experts, decay=envs.VLLM_MOE_CACHE_DECAY)
    return LFRUPolicy(num_experts)


class LFRUPolicy:
    """``score = freq / age`` from the entry's counters. The default.

    The provider already maintains ``freq`` and ``last`` inside its residency
    entries; this policy reads them and keeps no state of its own, so it is
    exactly the expression the victim scans used inline.
    """

    name = "lfru"

    def __init__(self, num_experts: int) -> None:
        self._num_experts = num_experts

    def on_hit(self, expert_id: int) -> None:
        pass

    def on_insert(self, expert_id: int) -> None:
        pass

    def on_evict(self, expert_id: int) -> None:
        pass

    def score(self, expert_id: int, freq: int, last: int, clock: int) -> float:
        return freq / (clock - last + 1)


class EWMAPolicy(LFRUPolicy):
    """Additive score over statistics that survive eviction.

        score(e) = w_f * ewma[e] * decay^age + w_r * exp(-age / tau)
                 + w_m * prior[e]

    ``ewma`` decays per event (the tier's clock, one tick per hit or insert),
    applied lazily at read time. ``decay=1.0`` never forgets and wins on
    stationary routing but loses after workload shifts; 0.999 was the only
    tested value that lost on neither (see bench/hit_rate_sweep.py).

    ``prior`` is the manifest seam: a table loaded at boot that biases
    residency toward experts that were hot in a previous run. Empty until
    seeded; entirely advisory -- it adds to the score and never pins.
    """

    name = "ewma"

    # Recency term, hardcoded off: measured contribution on synthetic traces
    # was ~0 against the frequency term (lfu-persist == ewma within noise).
    # Kept in the score shape so a real-trace sweep can turn it on without a
    # format change.
    W_R = 0.0
    TAU = 64.0

    def __init__(self, num_experts: int, decay: float = 0.999) -> None:
        super().__init__(num_experts)
        self.decay = decay
        self.w_f = 1.0
        self.w_m = 1.0
        self._ewma: dict[int, float] = {}
        self._last: dict[int, int] = {}
        self._clock = 0
        self.prior: dict[int, float] = {}

    def _bump(self, expert_id: int) -> None:
        self._clock += 1
        last = self._last.get(expert_id)
        gap = self._clock - last if last is not None else 0
        self._ewma[expert_id] = self._ewma.get(expert_id, 0.0) * self.decay**gap + 1.0
        self._last[expert_id] = self._clock

    def on_hit(self, expert_id: int) -> None:
        self._bump(expert_id)

    def on_insert(self, expert_id: int) -> None:
        self._bump(expert_id)

    def score(self, expert_id: int, freq: int, last: int, clock: int) -> float:
        # freq/last/clock are the entry's LFRU bookkeeping; this policy runs
        # on its own clock and ignores them.
        age = self._clock - self._last.get(expert_id, self._clock)
        s = self.w_f * self._ewma.get(expert_id, 0.0) * self.decay**age
        if self.W_R:
            s += self.W_R * math.exp(-age / self.TAU)
        if self.prior:
            s += self.w_m * self.prior.get(expert_id, 0.0)
        return s
