# MoE Expert Weight Caching

vLLM can run MoE models that exceed available GPU memory by keeping all expert
weights in CPU pinned memory and caching only the most-recently-used
experts in a fixed-size GPU scratch buffer.

For unquantized checkpoints, expert weights are loaded straight into CPU
pinned memory, so peak GPU usage never includes them. FP8 checkpoints
currently load expert weights onto the GPU first and offload after
`process_weights_after_loading`, so the *load* step still needs GPU capacity
for the full model; lifting that is follow-up work.

| Option | Default | Description |
| --- | --- | --- |
| `--moe-expert-cache-size N` | `0` (disabled) | Number of expert slots to allocate in the GPU buffer per layer |
| `--moe-expert-cache-split` | `token` | How to evaluate a forward that needs more experts than the cache holds: `token` or `expert` |
| `--moe-expert-cache-max-size N` | `0` (fixed) | Ceiling for a live resize of the cache (see *Live resizing*); scale buffers are allocated at this many rows once |

!!! note
    Expert caching is not compatible with expert parallelism (EP > 1),
    data parallelism, sequence parallelism, or LoRA. Backends are limited
    to the Triton and XPU MoE kernels (plus the vLLM CUTLASS fp8 kernel);
    backends that repack expert weights or ignore `expert_map` are
    rejected at startup.

## Quick start

```bash
# OLMoE-1B-7B: 64 experts, fits on 8 GB GPU with 16 cached per layer
vllm serve allenai/OLMoE-1B-7B-0924 \
    --moe-expert-cache-size 16
```

### Python API

`moe_expert_cache_size` is exposed as a direct `LLM` constructor parameter:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="allenai/OLMoE-1B-7B-0924",
    moe_expert_cache_size=16,
    moe_expert_cache_split="token",  # or "expert"
)
```

## Architecture (RFC #38256)

The cache is implemented as a `CachedWeightProvider` — the kernel does not
know or care where weights came from.

### How it works

```text
Everything fits (the common case, including every decode step):
  topk_ids -> provider.prepare():
    hit  -> bump the expert's frequency and recency
    miss -> evict the lowest-scoring expert, H2D copy into its slot
  -> kernel.apply(result.w1, result.w2, expert_map=result.expert_map)
```

Eviction is LFRU rather than plain LRU: an entry scores `freq / age`, so a
rarely used expert loses to a frequently used one even if the latter was
touched longer ago. Pure LRU behaves badly here, because MoE layers run in
sequence and early layers always look recently used.

The provider hands the kernel an `expert_map` — global expert id to buffer
slot, or `-1` for experts that are not resident — which is the same convention
expert parallelism uses, so `topk_ids` is passed through untouched. The map
lives in a GPU `int32` tensor with a pinned host mirror, rebuilt per call and
uploaded once.

### When the cache is smaller than a forward needs

A single forward can route to more distinct experts than the cache holds; a
batched prefill routinely routes to nearly all of them. `--moe-expert-cache-split`
picks how that forward is evaluated.

`token` (default) splits the batch by rows into chunks that fit, and
concatenates. Each token's full sum still happens inside one kernel call, so
**output is identical to running without the cache**. The cost is one kernel
launch per chunk, and as the cache approaches `top_k` no two tokens can share a
chunk — a long prefill can end up with one launch per token.

`expert` splits the expert set instead: run the whole batch once per group of
at most `--moe-expert-cache-size` experts, with the map hiding the rest, and
sum the results. That is `ceil(experts_used / cache_size)` launches whatever
the batch size, and each expert is fetched at most once per forward instead of
being refetched as the cache thrashes. **Output differs from the uncached path
at rounding level**, because each group's partial sum is rounded to the model
dtype before being accumulated.

Either way the cache must hold at least `top_k` experts, checked at startup.
For the `token` split that is a hard floor (one token's experts have to be
resident together); the `expert` split could in principle go lower, since it
sums partial results across groups — the shared floor is a deliberate
simplification, not a mathematical limit.

## CUDA graphs

The cache is compatible with piecewise CUDA graphs, which is the default
when `--moe-expert-cache-size` is set without `--enforce-eager`. vLLM adds
the MoE op (`vllm::moe_forward`) to `splitting_ops` and caps
`cudagraph_mode` at `PIECEWISE`: every non-MoE segment of the model is
captured and replayed, while the MoE layer — routing, cache management,
H2D copies, and the grouped GEMM — runs eagerly between segments. The
cache's GPU buffers and its `expert_map` are allocated once and updated
in place, so captured segments never observe a stale address.

All piecewise graphs draw from vLLM's one process-wide graph memory pool
(`Platform.get_global_graph_pool()`), and the MoE op copies its output
into a persistent buffer outside that pool (`_maybe_stabilize_output`)
before the next captured segment reads it. Per-graph private pools --
which can consume gigabytes across a few hundred captures -- are never
used on this path.

Full-graph capture (`FULL`, `FULL_AND_PIECEWISE`) is not supported: a
capture would freeze one forward's cache state into every replay.
`--enforce-eager` remains available and is required when compilation is
disabled (`-O0`).

## Known costs

Each MoE layer performs one device sync per forward to read the routing
decision onto the host (`topk_ids.unique()`); this is the latency floor of
the current design. Overlapping it with a routing-ahead prefetch is
planned as a follow-up (see RFC #38256).

When a disk tier is configured, the reads a `prepare()` call needs are
issued to a small pool of reader threads and drained as they complete, so
one read is in flight while the previous one's H2D copy and bookkeeping
happen. `VLLM_MOE_DISK_IO_THREADS` sets the pool size (default 2, capped
at 4) and `VLLM_MOE_DISK_PIPELINE=0` restores serial reads. Two readers
is a measured choice, not a conservative default: on NVMe, expert-sized
`O_DIRECT` reads into pinned memory need a second in flight to reach the
device's ceiling, while wide fan-out costs tail latency without adding
throughput.

`VLLM_MOE_DISK_PREFETCH=1` additionally starts the next expert group's
RAM fills under the current group's kernel in split forwards (requires
`VLLM_MOE_RAM_CACHE` at least twice the GPU capacity, for eviction
slack). It is off by default on measurement, not caution: single-stream
prefill is NVMe-bandwidth-bound and the in-plan pipeline already
overlaps within groups, so cross-group prefetch only reorders the same
reader queue. The lever exists for kernel-heavy configurations where
per-group compute exceeds per-group read time.

Pinned allocations for the RAM tier and the full-DRAM mirrors are
page-locked at their exact size (cudaHostRegister) rather than through
torch's caching host allocator, whose power-of-two buckets waste up to
~40% on non-power-of-two expert tensors, and every large page-lock is
budget-checked against available host memory first -- pinning near the
free-RAM scale livelocks the host rather than failing cleanly.

## Zero copy on unified memory

On a discrete GPU an expert has to be copied into device memory before a
kernel can read it. On unified-memory boxes (GB10 and friends) host and
device memory are one physical pool, so a page-locked RAM-tier row already
carries a device address and that copy is pure overhead -- measured at
1.35 TB of host-to-device traffic over a single 256-token run.

`VLLM_MOE_ZERO_COPY=1` points the MoE kernel straight at the disk tier's
pinned RAM pool. The GPU slot tier disappears: `--moe-expert-cache-size`
effectively becomes `VLLM_MOE_RAM_CACHE`, residency collapses to one level,
and no fill happens at all. The kernel takes the expert dimension's stride
explicitly and only requires `stride(-1) == 1`, so a strided view over the
record pool is a legal weight tensor.

Two costs to weigh against the machinery it deletes:

- Reading registered host memory is slower per GEMM than reading device
  memory (measured ~150 vs ~210 GB/s on GB10), so the win depends on how
  much of the run was fill and per-group bookkeeping rather than compute.
  It is largest at high concurrency, where a batch's expert union forces
  several groups per layer.
- The mode is plain-record only. FP8 records would still have to be
  dequantized into a device buffer, which is the fill under another name;
  configuring both raises at startup, as does combining it with
  `VLLM_MOE_DISK_PREFETCH` (a prefetch writes a slot outside `prepare()`,
  where the kernel-read protocol cannot see it).

Because the kernel, not an H2D copy, is now a slot's last reader, slot
reuse is ordered against `prepare()` generations: each call records one
event covering every slot it exposed, and a disk read waits on the
generation after the slot's last exposure before overwriting it.

At engine start the cache warns when `top_k x max_num_seqs` exceeds the
GPU capacity: a batch whose expert union does not fit chunks and
refetches every step, which serving throughput pays for directly.

## Observability

### DEBUG-level hit/miss log

Set `VLLM_LOGGING_LEVEL=DEBUG` to get a per-layer running hit/miss total,
logged every 1000 `prepare()` calls:

```text
DEBUG vllm...expert_weight_provider: Expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

Read the hit rate as a locality signal only under `--moe-expert-cache-split
token`. Under `expert`, a forward wider than the cache walks the whole expert
set group by group, so the rate mostly reports how often that happened rather
than how well the cache is sized — and the eviction policy has little to do
either, since each group displaces the last.

## Sizing guidance

Set `--moe-expert-cache-size` to the number of experts that must fit on
GPU simultaneously per layer. For a model with `E` experts and `top_k`
routing:

- **Minimum useful**: `top_k` (one slot per active expert per token, no
  eviction during decode)
- **Typical decode**: `2 * top_k` – `4 * top_k` gives headroom for
  locality without wasting VRAM
- **Maximum** (no-op): `E` (all experts on GPU, equivalent to normal mode)

Below roughly `E / 2` a batched prefill will start needing more experts than
fit, and `--moe-expert-cache-split` starts to matter. Keep the default `token`
unless prefill latency is the problem; switch to `expert` when it is, and
verify the quality impact for your model rather than assuming it is negligible.

## Live resizing (MV-WSA)

One GPU byte is a KV block or an expert slot, never both, and by default the
split is frozen at startup. With `--moe-expert-cache-max-size N` (N >= the
cache size) every Triton-backed layer's slot buffer may be resized while
serving, between the split's floor (`top_k` under the token split, 1 under
the expert split) and N; per-expert scale buffers are allocated at N rows
once and never move, which is why only Triton backends qualify (CUTLASS fp8
asserts scale rows equal weight rows, XPU captures the buffers once). A
ceiling on a layer that cannot resize is logged and ignored.

`VLLM_MOE_MVWSA=expert-union|kv-peak` turns the controller on. At each
busy-to-idle transition of the engine loop, with no request holding a KV
block, it reads the widest per-forward expert union of the epoch from the
workers, the peak live KV block count plus the prefix cache as KV demand,
and the preemption delta as KV pressure, and re-splits one constant byte
budget (the startup split's cost) between the two pools:

- `expert-union`: the cache gets exactly its observed working set, never
  more; KV gets what is left, but never less than its observed demand plus
  15% headroom (the live pool under pressure).
- `kv-peak`: KV is sized to its observed peak plus headroom and the experts
  get the rest -- the WiSP rule, kept for comparison.

A move rebuilds the KV tensors (they come back zeroed, so the prefix cache
is reset) and reallocates each layer's slot buffer, in the memory-safe
order: whichever side gives bytes up moves first. Measured on the GB10 with
Qwen3-30B-A3B over an fp8 disk store, a 48-layer move costs 0.4 s (shrink)
to 1.1 s (grow), and greedy output is unchanged across 32 -> 48 -> 16 -> 64.
Hysteresis: a target within 2 slots of the current capacity is ignored, a
new target must survive two consecutive barriers, and epochs shorter than 8
steps are not acted on. `VLLM_MOE_MVWSA_LOG=path` writes every observation,
decision and move as JSONL.

Not supported with the controller on: speculative decoding, KV connectors,
sleep mode, full CUDA graphs (piecewise is required anyway), zero copy, and
models with pinned sink blocks.

## GPU memory note

Expert weights in CPU pinned memory are invisible to the `--gpu-memory-utilization`
profiler. The profiler will underestimate available KV cache headroom by the
expert weight footprint (a safe margin, not a hazard), but exact
`gpu-memory-utilization`-based sizing will be off.

## Tests

```bash
# Unit tests: CachedWeightProvider
pytest tests/kernels/moe/test_expert_lru_cache.py -v
```
