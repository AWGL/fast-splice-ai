# FastSpliceAI

Accelerated CPU inference for the SpliceAI splice-variant models: same weights,
same numbers, **~6x faster**, and an install that no longer pulls in TensorFlow.

Output is byte-identical to SpliceAI v1.3.1 — verified over 726 variants against
the original running on TensorFlow. See [Validation](docs/validation.md).

## Quick start

Two things are needed that this repo cannot ship: an indexed **GRCh38 FASTA**,
and the **SpliceAI model weights** (`spliceai1.h5` .. `spliceai5.h5`, Illumina's,
CC BY-NC 4.0 — they come with the original SpliceAI repo).

MANE v1.5 transcript tables are checked in, so there is nothing to download or
build:

```sh
git clone https://github.com/your-org/fastspliceai.git
cd fastspliceai
pip install -r requirements.txt

python examples/grch38_example.py \
    /path/to/hg38.fa \
    data/mane_v1.5/annotation.txt.gz \
    data/mane_v1.5/genepred.txt.gz \
    /path/to/spliceai/models
```

```
#CHROM       POS REF ALT                  SYMBOL  DS_AG  DS_AL  DS_DG  DS_DL  DP_DL Any_splicing_aberration Partial_exon_deletion
     9 128504906   T   G RefSeqTx-NM_001003722.2   0.00   0.01   0.42   0.99     -2                     YES                   YES
     9  21994137   A   C    RefSeqTx-NM_058195.4   0.02   0.05   0.52   0.99      2                     YES                   YES
```

That is the whole example — 27 lines, no VCF in or out. It scores two variants
through the Python API and prints the delta scores plus the SAI-10k-calc
aberration calls.

Both variants disrupt a canonical `GT` donor, and between them they show why the
transcript set matters:

* **GLE1** `9-128504906-T-G`. Exon 1 ends at chr9:128,504,904 and the reference
  reads `GCGAG|GTGAGC`, so this is the `T` of the donor. `DS_DL 0.99` at
  `DP_DL -2` — near-certain loss, at exactly that base.
* **CDKN2A p14ARF** `9-21994137-A-C`. Exon 1β ends at chr9:21,994,139 on the
  minus strand, reading `CAG|GTAGG`, so this is the donor +2 base. It lies
  **outside the MANE Select transcript** (p16INK4a, `NM_000077.5`) altogether —
  with a Select-only annotation it produces no row at all. See
  [Transcripts](docs/transcripts.md#mane-select-and-mane-plus-clinical).

Nothing is discovered implicitly: the reference, both transcript tables and the
model directory are all passed in. To use your own transcripts instead of MANE,
build the two tables with [`fastspliceai-tx`](#2-build-annotation-tables-from-your-own-transcripts)
and pass those paths.

## Install

```sh
pip install "git+https://github.com/your-org/fastspliceai.git"
```

Pin a tag for anything reproducible:

```sh
pip install "git+https://github.com/your-org/fastspliceai.git@v0.1.0"
```

> Replace `your-org/fastspliceai` with the real path once the repo is published.

That gives you four commands — `fastspliceai` (scoring), `fastspliceai-tx`
(annotation tables from a GFF), `fastspliceai-sai10k` (aberration calls) and
`fastspliceai-serve` (HTTP API) — and about 40 MB of dependencies. There is no TensorFlow
and no Keras: the `.h5` weights are read with `h5py` and converted to ONNX on
first use, cached under `$XDG_CACHE_HOME/fastspliceai`.

The **model weights are not bundled**. They are Illumina's, under CC BY-NC 4.0.
Point `--models` at a directory holding `spliceai1.h5` .. `spliceai5.h5`; it is
required, and the weights are never discovered implicitly, so every run records
which copy it used.

Every dependency is a plain wheel, so a stock `venv` is enough — see
[Running in production](docs/production.md) for air-gapped installs, conda and
version pinning.

### Development

Install the dependencies only — the package itself stays where you are editing
it, uninstalled:

```sh
git clone https://github.com/your-org/fastspliceai.git
cd fastspliceai
pip install -r requirements-dev.txt

pytest
```

`pytest` needs no setup: `pyproject.toml` sets `pythonpath = ["src"]`, so the
tests import the working tree directly.

The console scripts (`fastspliceai`, `fastspliceai-tx`, `fastspliceai-sai10k`,
`fastspliceai-serve`) only exist once the package is installed, so from a bare checkout call the
modules and point Python at `src/`:

```sh
export PYTHONPATH=$PWD/src

python -m fastspliceai.cli -I in.vcf -O out.vcf -R hg38.fa -A grch38
python -m fastspliceai.transcripts annotation.gff3.gz -o ann.txt --out-genepred gp.txt
python -m fastspliceai.sai10k.cli -I scored.vcf -t gp.txt -O calls.tsv
```

`requirements.txt` mirrors `[project.dependencies]` in `pyproject.toml`;
`requirements-dev.txt` adds pytest.

## Usage

Three commands. Only the first is needed to score variants.

### 1. Score a VCF

```sh
fastspliceai -I in.vcf -O out.vcf -R hg38.fa -A grch38 -D 500 \
    --models /path/to/spliceai/models
```

**On `-j`:** the five ensemble models already run concurrently across all cores
within one process, so extra workers mostly contend. Measured on 12 logical
cores, 200 variants:

| | variants/s | vs stock |
|---|---|---|
| `-j 1` | **4.41** | **5.96x** |
| `-j 2` | 3.57 | 4.82x |
| `-j 5` | 3.48 | 4.70x |

`-j 1` is the default and the right answer on a typical workstation. Processes
only help once you have substantially more cores than the ~15 threads a single
process already uses — and each one costs another copy of the models.

| flag | meaning |
|---|---|
| `-R` | reference FASTA (indexed) — **required** |
| `-A` | `grch37`, `grch38`, or your own annotation table — **required** |
| `-m` | directory holding `spliceai1.h5` .. `spliceai5.h5` — **required**, never inferred |
| `-D` | window each side of the variant, **default 500**. Upstream SpliceAI uses 50; see [Performance](docs/performance.md#window-size-is-nearly-free) |
| `-M` | mask annotated gains / unannotated losses (0 or 1, default 0) |
| `-t` | inference threads. Default is **half the CPU allocation, minimum 4**, so it shares a host politely. Also settable as `FASTSPLICEAI_CPUS` |
| `-j` | worker processes, **default 1**. Leave it there unless you have many cores — see below |
| `--raw-scores` | also emit the REF/ALT probabilities behind each delta score |
| `--chunk-size` | records held in memory at once (default 500) |

Reads and writes VCF through pysam, streaming in chunks, so `-` works for stdin
and stdout:

```sh
bcftools view region.bcf | fastspliceai -R hg38.fa -A grch38 | bgzip > scored.vcf.gz
```

### 2. Build annotation tables from your own transcripts

A one-off step. MANE is the sensible default for clinical work:

```sh
curl -O https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.refseq_genomic.gff.gz

fastspliceai-tx MANE.GRCh38.v1.5.refseq_genomic.gff.gz \
    -o annotation.txt --out-genepred genepred.txt \
    --name-field transcript --name-prefix RefSeqTx-
```

That writes **both** tables from one file, so scoring and classification cannot
end up on different transcript sets. Takes ~30 s; keep the outputs and reuse
them. Then score with `-A annotation.txt`.

GFF3 and GTF are both accepted, including NCBI RefSeq, GENCODE and Ensembl
dialects. See [Transcripts](docs/transcripts.md) for where to download one, why
MANE Plus Clinical matters, and how to update.

### 3. Classify the result (SAI-10k-calc)

```sh
fastspliceai -I in.vcf -O scored.vcf -R hg38.fa -A annotation.txt -D 500 --raw-scores
fastspliceai-sai10k -I scored.vcf -t genepred.txt -O calls.tsv
```

Predicts the splicing aberration type — pseudoexon activation, intron retention,
exon skipping, partial exon deletion and so on. Pure Python, no R. See
[SAI-10k-calc](docs/sai10k.md).

## Server

Start-up costs ~3 s (imports, five models, transcript tables) against ~0.5 s to
score a variant, so an on-demand lookup is mostly overhead. Run a warm engine
instead:

```sh
fastspliceai-serve -R hg38.fa -A ann.txt -t gp.txt -m /path/to/spliceai/models \
    --cache ~/.cache/fastspliceai/scores.db

curl -s localhost:8722/health
curl -s localhost:8722/score -d '{"variants": ["chr9-128504906-T-G"]}'
```

| | per query |
|---|---|
| cold CLI invocation | ~3 s |
| warm server | ~0.5 s |
| warm server, cache hit | ~1 ms |

Two endpoints: `POST /score` takes `{"variants": [...]}` and returns the rows as
JSON; `GET /health` reports the loaded transcript count, `-D`, and cache
statistics, so you can confirm what a running instance is actually using.

Requests are served one at a time. ONNX Runtime already uses every core for a
single request, so concurrent requests would only contend - the same reason
batching is a pessimisation.

> **No authentication, no TLS, no audit log, no rate limiting.** It binds
> `127.0.0.1` deliberately. This is a local accelerator, not a service to
> expose; put it behind infrastructure you already trust before it handles
> patient data.

### As a container

`Containerfile` plus a systemd Quadlet unit are in `podman/`:

```sh
podman build -t fastspliceai:latest .
cp podman/fastspliceai.container podman/fastspliceai-cache.volume ~/.config/containers/systemd/
systemctl --user daemon-reload && systemctl --user start fastspliceai
```

The reference genome and model weights are mounted read-only rather than baked
in — the weights are Illumina's under CC BY-NC 4.0 and must not be redistributed
in an image. See [podman/README.md](podman/README.md).

### Caching

`--cache` stores scored variants in SQLite. A score never changes for a fixed
reference, transcript set, model weights, `-D` and `-M`, so it is safe to keep —
**provided a stale entry can never be served.**

Every key is salted with a fingerprint of those inputs (file sizes and mtimes
for the tables and model files, plus the distance and mask). Rebuild your
annotation, swap the weights, or change `-D`, and the fingerprint changes:
older rows stop matching rather than being returned against the wrong
transcripts. `/health` reports how many stale rows are present, and
`--purge-stale` removes them.

Variants that score nothing — outside every transcript, reference mismatch —
are cached as an empty result, so they are not rescored on every request.

## Python API

```python
from fastspliceai import SpliceAI

sai = SpliceAI(reference="hg38.fa", models="/path/to/spliceai/models",
               gff="MANE.GRCh38.v1.5.refseq_genomic.gff.gz", distance=500)

df = sai.score(["9-128504906-T-G", ("17", 43045802, "G", "A")])
```

Returns a `pandas.DataFrame`: one row per (variant, transcript, allele) with
delta scores, the REF/ALT probabilities behind them, and the SAI-10k-calc
aberration calls. Build the object once and reuse it. Full reference in
[docs/api.md](docs/api.md).

## Defaults worth knowing

| | default | why |
|---|---|---|
| `-D` distance | **500** | 12% slower than 50, sees ten times as much sequence. Upstream SpliceAI uses 50; precomputed score files are 50-only, which is a real source of missed variants. |
| threads | **half the CPU allocation, minimum 4** | This is expected to share a host. Detection respects cgroups, `cpuset`, SLURM and `OMP_NUM_THREADS`, so a container limited to 2 CPUs uses 2, not the host's 64. |
| processes (`-j`) | **1** | The five models already run concurrently across all cores in one process. |
| annotation | **none — required** | There is no default build. Scoring GRCh38 data against a GRCh37 table returns empty results rather than failing, so it must be explicit. |
| model weights | **none — required** | Never discovered implicitly, so a run always records which weights it used. |

All of them are overridable. `FASTSPLICEAI_CPUS` caps threads everywhere —
CLI, API and server — which is the knob to set when embedding this in a
webserver.

## Testing

```sh
pytest                    # everything that can run on this machine
pytest -m "not slow"      # skip the ones needing many inference passes
```

Tests that need data the repo cannot ship — a GRCh38 reference, the Illumina
model weights, an hg19 fixture — skip themselves with a reason rather than
failing. `tests/make_hg19_fixture.py` builds the one fixture that is fetchable.

Four layers, each catching a different kind of mistake:

| file | what it protects |
|---|---|
| `test_scores.py` | the published upstream vectors, byte for byte |
| `test_bulk.py` | 200 variants against stock SpliceAI output |
| `test_invariants.py` | known-pathogenic variants keep scoring highly; results do not depend on process count |
| `test_sai10k.py`, `test_protein.py` | 22 classification columns and the protein sequence, against the R implementation |
| `test_api.py` | argument handling, symbolic alleles, the single-vs-many variant ambiguity |

`test_invariants.py` exists because of a specific failure. An INT8 path was
calibrated on random sequence; every strong call collapsed to noise; the whole
suite stayed green, because nothing asserted that a variant whose biology is not
in doubt still scores highly. Golden files only catch changes to the exact rows
they contain. A canonical `GT` donor disruption that must keep scoring above 0.9
survives refactors that golden files do not.

## Documentation

| | |
|---|---|
| [Validation](docs/validation.md) | evidence it matches stock SpliceAI exactly |
| [Performance](docs/performance.md) | throughput, window size, internals, roadmap |
| [Transcripts](docs/transcripts.md) | annotation sources, MANE Plus Clinical, updating |
| [SAI-10k-calc](docs/sai10k.md) | splicing aberration classification |
| [Running in production](docs/production.md) | offline installs, reproducibility, memory, serving |
| [Python API](docs/api.md) | full API reference |
| [Container deployment](podman/README.md) | podman image and systemd unit |
| [MANE tables](data/mane_v1.5/README.md) | what the bundled transcript tables are |

## Why this exists

SpliceAI is a fully convolutional network with a ±5000 nt receptive field. The
stock tool runs it on a 10,101 nt window to produce **101** scored positions,
once per model, twice per variant (reference and alternate), five models deep:

| | |
|---|---|
| MACs per output position | 693 k |
| MACs per forward pass (10,101 nt window) | 7.0 G (14.0 GFLOP) |
| Passes per biallelic SNV | 10 (5 models × ref/alt) |
| **MACs per variant** | **70 G (140 GFLOP)** |
| Useful output positions | 101 |

Roughly 99% of the arithmetic is discarded. FastSpliceAI keeps the model
untouched and fixes the scheduling and the runtime around it.

## Licensing

SpliceAI's source was **GPLv3 through v1.3.1**; Illumina relicensed the upstream
repository to PolyForm Strict 1.0.0 on 2025-07-18, which forbids distributing
the software or making new works based on it. FastSpliceAI therefore derives
from the GPLv3-era code and is distributed under the **GPLv3**.
