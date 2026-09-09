# fast-splice-ai

Accelerated CPU inference for the SpliceAI splice-variant models.

Output is identical to SpliceAI v1.3.1 — verified over 726 variants against
the original running on TensorFlow. See [Validation](docs/validation.md).

## Quick start

Two things are needed - an indexed **GRCh38 FASTA**,
and the **SpliceAI model weights** (`spliceai1.h5` .. `spliceai5.h5` -  they come with the original SpliceAI repo).

MANE v1.5 transcript tables are included.

```sh
git clone https://github.com/AWGL/fast-splice-ai.git
cd fast-splice-ai
pip install -r requirements.txt

python examples/grch38_example.py \
    /data/resources/reference_genomes/GRCh38_full_analysis_set_plus_decoy_hla_masked.fa \
    data/mane_v1.5/annotation.txt.gz \
    data/mane_v1.5/genepred.txt.gz \
    ../spliceai/models

```
Note ../spliceai/models is here https://github.com/Illumina/SpliceAI/tree/master/spliceai/models
```
=== SpliceAI delta scores ===
  gene                   #CHROM       POS REF ALT  DS_AG  DS_AL  DS_DG  DS_DL
  GLE1  donor +2 canonical GT   chr9 128504906   T   G   0.00   0.01   0.42   0.99
   NF1  c.288+2025T>G          chr17  31161118   T   G   0.81   0.00   0.87   0.00
  CFTR  c.3718-2477C>T          chr7 117639961   C   T   0.10   0.00   0.16   0.00
  ...

=== predicted consequence (SAI-10k-calc + protein) ===
  GLE1  Partial exon deletion (41bp) + premature stop at codon 54, NMD predicted
   NF1  Pseudoexon activation in intron 3 (108bp) + premature stop at codon 97, NMD predicted
  CFTR  No splicing aberration predicted
```

## Install

```sh
pip install "git+https://github.com/AWGL/fast-splice-ai.git"
```

### Development

Install the dependencies only

```sh
git clone https://github.com/AWGL/fast-splice-ai.git
cd fast-splice-ai
pip install -r requirements.txt

pytest
```

## Usage

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


### As a container

`Containerfile` plus a systemd Quadlet unit are in `podman/`:

```sh
podman build -t fastspliceai:latest .
cp podman/fastspliceai.container podman/fastspliceai-cache.volume ~/.config/containers/systemd/
systemctl --user daemon-reload && systemctl --user start fastspliceai
```

The reference genome and model weights are mounted read-only rather than baked
in.

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

### SAI-10k-calc thresholds

Match the R implementation except where noted. Override with
`SpliceAI(thresholds={"DS_ALDL_MIN_T": 0.1})`, `Thresholds.strict()`, or the
`--DS_*` flags on `fastspliceai-sai10k`.

| | default | |
|---|---|---|
| `AG_T` / `DG_T` | 0.2 | acceptor / donor gain needed for cryptic activation |
| `AGDG_T` | 0.0 | floor on the larger of the two gains |
| `DS_GL_T` | -0.2 | gain-minus-loss margin |
| `DS_AGDG_MIN_T` / `_MAX_T` | 0.02 / 0.2 | both gains, for a predicted gained exon |
| `DS_ALDL_MIN_T` / `_MAX_T` | **0.05** / 0.2 | both losses, for exon skipping and intron retention |
| `GEX_size_MIN` / `_MAX` | 25 / 500 bp | plausible pseudoexon size |

**`DS_ALDL_MIN_T` is the one that differs** — SAI-10k-calc uses 0.02, which lets
a 0.05 acceptor-loss score (noise) help support an intron-retention call. Pass
`Thresholds(DS_ALDL_MIN_T=0.02)` to reproduce the R tool exactly, or
`Thresholds.strict()` to raise `AGDG_T`, `DS_AGDG_MIN_T` and `DS_ALDL_MIN_T` to
0.1.

Every call carries `Event_confidence` and `Confidence_note` with the limiting
score, so a marginal call stays visible whatever the thresholds.

## Testing

```sh
pytest                    # everything that can run on this machine
pytest -m "not slow"      # skip the ones needing many inference passes
```

## Documentation

| | |
|---|---|
| [Validation](docs/validation.md) | evidence it matches stock SpliceAI exactly |
| [Performance](docs/performance.md) | throughput, window size, internals, roadmap |
| [Transcripts](docs/transcripts.md) | annotation sources, MANE Plus Clinical, updating |
| [SAI-10k-calc](docs/sai10k.md) | splicing aberration classification |
| [Python API](docs/api.md) | full API reference |
| [Container deployment](podman/README.md) | podman image and systemd unit |
| [MANE tables](data/mane_v1.5/README.md) | what the bundled transcript tables are |


## Licensing

SpliceAI's source was **GPLv3 through v1.3.1**; Illumina relicensed the upstream
repository to PolyForm Strict 1.0.0 on 2025-07-18, which forbids distributing
the software or making new works based on it. fast-splice-ai therefore derives
from the GPLv3-era code.
