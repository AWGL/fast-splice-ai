# Performance

Where the speed comes from, what it costs, and what is worth doing next.

## Throughput

Measured head-to-head against stock SpliceAI v1.3.1 (TensorFlow 2.21 with
oneDNN, same five `.h5` weight files, same machine, 10 cores), 200 variants at
`-D 50`:

| | variants/s | speedup |
|---|---|---|
| stock SpliceAI (TensorFlow) | 0.70 | 1.00x |
| FastSpliceAI `-j 1` | 2.96 | **4.21x** |
| FastSpliceAI `-j 10` | 4.00 | **5.69x** |

All 200 records - 122 SNVs, 42 insertions, 36 deletions - are **byte-identical**
to stock. `-j 1` and `-j 10` also agree exactly, so results do not depend on how
the work is sharded.

Projected once the incremental ALT pass lands (MAC ratios are exact; the
variants/s figures are de-rated by the measured MAC-to-wall-clock efficiency of
the numpy prototype):

| | G MAC/variant | vs stock | proj. variants/s `-j10` |
|---|---|---|---|
| stock SpliceAI | 69.5 | 1.00x | 0.70 |
| FastSpliceAI today | 45.2 | 1.54x | 4.00 |
| + incremental ALT | 28.8 | 2.42x | ~4.8 (~7x) |
| + tiled REF, clustered variants | 6.2 | 11.24x | ~25 (~35x) |

The tiled figure assumes variants dense enough for the 10 kb context to
amortise across a tile - panel or gene-wide rescoring, not a handful of
scattered variants.

## Widening the window is nearly free

Illumina's precomputed SNV scores are generated at `-D 50`, so they only report
splice changes within 50 bp of the variant. Effects further out are simply
absent from those files - which is a real source of missed variants, and the
main reason to rescore rather than look up.

Rescoring wider costs far less than it looks, because the fixed 10 kb context
dominates the window:

| `-D` | window | ms/pass | vs `-D 50` | bases scored per variant |
|---|---|---|---|---|
| 50 | 10,101 | 34.4 | 1.00x | 101 |
| 100 | 10,201 | 35.1 | 1.02x | 201 |
| 500 | 11,001 | 38.6 | **1.12x** | **1,001** |
| 1000 | 12,001 | 43.7 | 1.27x | 2,001 |
| 2000 | 14,001 | 53.7 | 1.56x | 4,001 |
| 4999 | 19,999 | 83.8 | 2.43x | 9,999 |

`-D 500` scores ten times as many positions for 12% more time. Per scored base,
`-D 4999` is about 40x cheaper than `-D 50`. FastSpliceAI at `-D 4999` still
runs faster than stock SpliceAI at `-D 50` while covering ~99x more sequence.

Precomputed lookups remain the right answer when `-D 50` genuinely suffices and
the variants are SNVs.

## Where the speed comes from

Measured on a 10-core i7-1355U (AVX2 + VNNI, no AVX-512/AMX), best-of-N.
Reproduce with `python benchmarks/bench.py`.

### 1. One shared thread pool — 13×

The largest single win, and a pure bug fix. Five models means five ONNX Runtime
sessions, and by default each session owns a thread pool whose idle workers
*spin*. Five spinning pools on one machine oversubscribe every core:

| | ms per forward pass |
|---|---|
| five sessions, per-session pools | 634 |
| five sessions, one global pool | **36** |

`Ensemble` calls `set_global_thread_pool_sizes` once and opens every session
with `use_per_session_threads = False`, falling back to disabling the spin.

### 2. Valid-convolution rewrite — 1.54× fewer MACs, exact

`convert.py` rewrites every `padding='same'` convolution to `padding='valid'`.
Each layer then computes only the positions that can still reach an output,
and the trailing `Cropping1D` becomes a no-op. Output is bit-identical
(max abs difference 1.0e-11).

This also makes the graph **length-agnostic**, which is what enables tiling.

### 3. Reference-pass caching — up to 1.5× per multi-allelic site

Upstream recomputes the reference pass inside its `for alt: for transcript:`
loop, so a tri-allelic site pays for the identical reference window three
times. FastSpliceAI keys `y_ref` on `(window, strand)`:

| site | passes (upstream → FastSpliceAI) | |
|---|---|---|
| biallelic SNV | 10 → 10 | 1.00× |
| tri-allelic | 30 → 20 | **1.50×** |

Honest note: a plain biallelic SNV gets nothing from this. Its gains come
entirely from §1, §2 and the runtime itself.

### 4. Tiled scanning — 50×, but only for dense workloads

Because the graph is length-agnostic, `Ensemble.tile()` scores a long region in
100 kb tiles, paying the 10 kb context once per tile instead of once per
position:

| | µs per scored base |
|---|---|
| one 10,101 nt window per variant | 1741 |
| 100 kb tiles | **35** |

**This only pays off when variants are dense enough to share a tile** — whole-gene
or panel-wide rescoring, all-possible-SNV precompute. For variants scattered
across the genome, which is the usual clinical case, it degenerates to the
per-variant path and buys nothing. It is available in the API; it is not part of
the scattered-variant story.

### 5. INT8 — 3.8×, but *not* exact

`--precision int8` gives a real 3.8× on VNNI hardware. It is off by default and
should stay off unless you re-validate.

Raw acceptor/donor probabilities match fp32 to <1e-4, which looks safe — but
delta scores are *differences* of two nearly identical predictions, so the
error does not cancel. Both upstream test vectors fail under INT8:

```
fp32:  C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193
int8:  C|TUBB8|0.15|0.26|0.00|0.06|89|-23|-500|193
                    ^^^^      ^^^^          ^^^^
```

Delta scores shift by up to ~0.02 and reported positions can flip. Keeping the
final stack in fp32 does not fix it. In the cases observed, position flips only
happened where the matching delta score was 0.00–0.01 (i.e. no signal), so the
practical impact is probably small — but you lose reproducibility against
published SpliceAI output.

## Roadmap

Ordered for the workload that actually matters here: **variants scattered across
the genome, not clustered**. Anything that relies on variants being near each
other is excluded, and the things already tested and rejected are listed so
nobody spends a week rediscovering them.

### Worth doing

**1. Incremental ALT pass.** The largest remaining exact win. Reference and
alternate differ in one base, so their activations are provably identical
outside a cone that grows 5 → 20 → 100 → 500 positions per layer. Recomputing
only that cone is **exact**, not an approximation — a numpy prototype reproduces
the full pass to 1.7e-10.

| ALT pass | G MAC | vs stock |
|---|---|---|
| stock `same` padding | 6.95 | 1.00× |
| valid rewrite (shipping) | 4.52 | 1.54× |
| coarse 4-stage incremental | 2.11 | 3.30× |
| per-layer incremental (prototyped) | 1.24 | **5.62×** |

Per variant that is 45.2 → 28.8 G MAC, about **1.4× wall clock** after de-rating
by the prototype's measured MAC-to-time efficiency. Entirely per-variant, so
scattering does not weaken it.

**2. Share the annotation table between workers.** Each `-j` worker loads its own
models and its own copy of the 20k-row transcript table: `-j10` costs ~1.8 GB
where `-j1` costs 181 MB. Memory, not compute, is what caps the useful process
count on a shared node, and `-j` is the main scaling knob for scattered
variants.

**3. Cheaper start-up.** Each process spends ~0.45 s re-parsing the annotation
table, which dominates the 0.68 s start-up and is paid again by every worker.
Caching a parsed form would mostly remove it — worth more than it sounds if you
run many small jobs rather than one big VCF.

### Bigger, but changes output

**4. Distil the five-model ensemble into one network.** Up to **5×**, and after
(1) it is the largest remaining lever for scattered variants. It changes
predictions, so it is a research project with its own validation burden, not a
patch.

**5. INT8.** A real 3.8×, already implemented behind `--precision int8`, but it
shifts delta scores by up to ~0.02 and can flip reported positions where the
score is ~0. Needs clinical revalidation.

**6. Newer hardware.** These numbers come from a laptop with AVX2 + VNNI, no
AVX-512 and no AMX. A Sapphire Rapids or newer server would be substantially
faster again on the INT8 and bf16 paths, for no code change.

### Tested and rejected

**Fusing the five models into one graph.** The idea was to run the ensemble as
five parallel branches in a single ONNX graph so ONNX Runtime could execute them
concurrently, rather than five sessions back to back. Built and measured:

| | ms/pass | speedup |
|---|---|---|
| five sequential sessions | 176 | 1.00× |
| fused graph, sequential execution | 177 | 1.00× |
| fused graph, parallel execution (inter-op 2) | 177 | 1.00× |
| fused graph, parallel execution (inter-op 5) | 176 | 1.00× |

No gain at any setting. Intra-op parallelism already saturates the cores at
these convolution sizes, so there is nothing for inter-op branches to reclaim.
The gain from `-j` comes from process isolation, not from extra parallelism.

**Tiled scanning.** `Ensemble.tile()` is ~50× per scored base, but only when
variants are dense enough to share a 100 kb tile. For scattered variants it
degenerates to the per-variant path. It stays in the API for whole-gene and
panel-wide scanning; it is not part of the scattered-variant story.

**Cross-variant reference caching.** Same reasoning. The cache still pays on
multi-allelic sites and on several transcripts at one locus, which is why it is
on by default, but it will rarely hit across unrelated variants.

**Batching variants into one inference call.** A pessimisation: ONNX Runtime
parallelises convolutions over the batch dimension, so batching halves the
spatial parallelism — 35 ms/pass at batch 1 versus 169 ms/pass at batch 2. Use
`-j` processes instead.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
