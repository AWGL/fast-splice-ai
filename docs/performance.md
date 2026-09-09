# Performance

## Throughput

Measured against stock SpliceAI v1.3.1 (TensorFlow with oneDNN), same five
`.h5` files, same machine (10-core i7-1355U), 200 variants at `-D 50`:

| | wall clock | variants/s | speedup |
|---|---|---|---|
| stock SpliceAI | 270.1 s | 0.74 | 1.00x |
| FastSpliceAI | 45.3 s | **4.41** | **5.96x** |

All 200 records byte-identical to stock. Reproduce with `benchmarks/bench.py`.

Rough planning figures for 100k variants, `-D 500`, on this laptop: about
**7 hours**. A server with more real cores scales close to linearly.

## Window size is nearly free

`-D` costs far less than it looks, because the fixed 10 kb receptive field
dominates the window:

| `-D` | window | ms/pass | vs `-D 50` | bases scored |
|---|---|---|---|---|
| 50 | 10,101 | 34.4 | 1.00x | 101 |
| 500 | 11,001 | 38.6 | **1.12x** | **1,001** |
| 1000 | 12,001 | 43.7 | 1.27x | 2,001 |
| 4999 | 19,999 | 83.8 | 2.43x | 9,999 |

`-D 500` is the default: ten times the window for 12% more time. Illumina's
precomputed score files are `-D 50` only, so effects further out are simply
absent from them — which is the main reason to rescore rather than look up.

Going wider is not free of consequences even though it is nearly free of cost.
Delta scores are a **maximum over the window**, so they only ever increase with
`-D`; the published 0.2 / 0.5 / 0.8 cutoffs were characterised at the standard
window. Filter on `DP_*` if you widen beyond 500.

## Where the speed comes from

**One shared thread pool, no spinning — 13x.** Five models means five ONNX
Runtime sessions, and by default each owns a pool whose idle workers spin. Five
spinning pools oversubscribe every core: 634 ms per forward pass against 36 ms
with spinning disabled.

**Concurrent ensemble execution — 1.12x.** The five models run on a thread pool
rather than one after another. A 32-channel convolution does not spread well
over many threads, but five independent models do, and ONNX Runtime releases
the GIL inside `run()`. Costs no extra memory, unlike processes.

**Valid-convolution rewrite — 1.54x fewer MACs, exact.** Every `padding='same'`
convolution becomes `padding='valid'`, so each layer computes only positions
that can still reach an output and the trailing `Cropping1D` becomes a no-op.
Bit-identical (max difference 1.0e-11), and it makes the graph length-agnostic.

**Reference-pass caching — up to 1.5x per multi-allelic site.** Upstream
recomputes the reference pass inside its `for alt: for transcript:` loop; a
tri-allelic site pays for the same window three times. Keyed on
`(window, strand)`.

**Tiled scanning — 50x per scored base, for dense workloads only.**
`Ensemble.tile()` pays the 10 kb context once per 100 kb tile instead of once
per variant: 1741 µs per scored base windowed, 35 µs tiled. This only helps
when variants are dense enough to share a tile — whole-gene or panel-wide
rescoring. For variants scattered across the genome it buys nothing.

## What does not help

Each of these was implemented and measured before being rejected.

| | result |
|---|---|
| **More processes** (`-j`) | 4.41 var/s at `-j 1`, 3.48 at `-j 5`. One process already saturates the cores; extra workers add start-up and contention. Only worth it when cores substantially exceed the ~15 threads one process uses. |
| **Batching variants** | 35 ms/pass at batch 1, 169 ms at batch 2. ONNX Runtime parallelises convolutions over the batch dimension, so batching halves the spatial parallelism. |
| **Fusing the five models into one graph** | 1.00x at every inter-op setting. Intra-op parallelism already saturates the cores at these convolution sizes. |
| **INT8 quantisation** | No speedup in a realistic configuration (37.7 s vs 36.4 s at `-j 5`) and up to 0.09 delta-score error. Removed. |

## What is left

**Distil the five-model ensemble into one network** — up to 5x, and the largest
remaining lever. It changes predictions, so it is a research project with its
own validation burden, not a patch. Note that simply *dropping* models is not a
substitute: 3 of 5 gives 1.67x for a worst-case delta-score error of 0.069.

**Incremental alternate pass** — reference and alternate differ in one base, so
their activations are provably identical outside a cone that grows 5 → 20 → 100
→ 500 positions per layer. Recomputing only that cone is exact; a numpy
prototype reproduces the full pass to 1.7e-10 and cuts the alternate pass from
4.52 to 1.24 G MAC. About **1.4x** overall in practice. Judged not worth it: it
needs intricate ONNX graph surgery in the numerical core, has a silent-wrongness
failure mode, and needs a full-computation fallback for indels, where reference
and alternate differ in length.

**Better hardware.** These numbers come from a laptop with AVX2 + VNNI, no
AVX-512 and no AMX, and 8 of its 10 cores are efficiency cores. Execution is
already at ~93% of this chip's peak sgemm throughput, and inference is 98% of
wall clock, so there is nothing left to win in *how* the arithmetic runs — only
in how much of it there is, or what it runs on.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
