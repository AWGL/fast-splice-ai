# FastSpliceAI

Accelerated CPU inference for the SpliceAI splice-variant models: same weights,
same numbers, **4-6x faster**, and an install that no longer pulls in TensorFlow.

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

That gives you three commands — `fastspliceai`, `fastspliceai-tx` and
`fastspliceai-sai10k` — and about 40 MB of dependencies. There is no TensorFlow
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

The console scripts (`fastspliceai`, `fastspliceai-tx`, `fastspliceai-sai10k`)
only exist once the package is installed, so from a bare checkout call the
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
fastspliceai -I in.vcf -O out.vcf -R hg38.fa -A grch38 -D 500 -j 4 \
    --models /path/to/spliceai/models
```

| flag | meaning |
|---|---|
| `-R` | reference FASTA (indexed) |
| `-A` | `grch37`, `grch38`, or your own annotation table |
| `-D` | window around the variant, default 50. **500 costs only 12% more and sees 10x as much** — see [Performance](docs/performance.md#widening-the-window-is-nearly-free) |
| `-M` | mask annotated gains / unannotated losses (0 or 1) |
| `-j` | worker processes (memory scales with this) |
| `--raw-scores` | also emit the REF/ALT scores behind each delta |
| `--precision` | `fp32` (exact, default) or `int8` (~3x faster, **not** exact) |
| `-m` | directory holding `spliceai1.h5` .. `spliceai5.h5` (**required**) |

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

## Documentation

| | |
|---|---|
| [Validation](docs/validation.md) | evidence it matches stock SpliceAI exactly |
| [Performance](docs/performance.md) | throughput, window size, internals, roadmap |
| [Transcripts](docs/transcripts.md) | annotation sources, MANE Plus Clinical, updating |
| [SAI-10k-calc](docs/sai10k.md) | splicing aberration classification |
| [Running in production](docs/production.md) | offline installs, reproducibility, memory |
| [Python API](docs/api.md) | full API reference |

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
