# Python API

For embedding in another application. Build the object once — it loads the five
models and the transcript tables — then reuse it; reference windows are cached
across calls.

```python
from fastspliceai import SpliceAI

sai = SpliceAI(
    reference="hg38.fa",
    annotation="data/mane_v1.5/annotation.txt.gz",   # required
    models="/path/to/spliceai/models",               # required
    transcripts="data/mane_v1.5/genepred.txt.gz",    # optional: turns on SAI-10k-calc
)

df = sai.score([
    "chr9-128504906-T-G",                        # chr-pos-ref-alt
    "chr9:21994137:A:C",                         # or colon separated
    ("chr8", 140300616, "T", "G"),               # or a tuple
    {"chrom": "chr17", "pos": 31161118, "ref": "T", "alt": ["G"]},
])
```

## Constructor

| argument | default | notes |
|---|---|---|
| `reference` | — | indexed FASTA |
| `annotation` | — | **required**: `"grch37"`, `"grch38"`, a table path, or a DataFrame. No default build — scoring GRCh38 against a GRCh37 table returns empty results rather than failing |
| `models` | — | **required**, never discovered implicitly |
| `transcripts` | `None` | genePred table; enables SAI-10k-calc and the protein prediction |
| `gff` | `None` | derive **both** tables from one GFF3/GTF instead of passing them separately |
| `distance` | `500` | window each side of the variant |
| `mask` | `0` | mask annotated gains / unannotated losses |
| `threads` | `None` | defaults to half the CPU allocation, minimum 4 |
| `thresholds` | `None` | a `Thresholds`, or a dict of overrides such as `{"DS_ALDL_MIN_T": 0.1}` |
| `protein` | `True` | predict the protein consequence of each event |

## What comes back

One row per (variant, transcript, allele), as a `pandas.DataFrame`:

```
#CHROM       POS REF ALT SYMBOL                   DS_AG DS_AL DS_DG DS_DL DP_DL
  chr9 128504906   T   G RefSeqTx-NM_001003722.2   0.00  0.01  0.42  0.99    -2

Consequence          Partial exon deletion (41bp) + premature stop at codon 54, NMD predicted
Event_confidence     moderate
Protein_consequence  DKG[CFRRMYVSSQAIFLFWMGGRARPTPYAGEPTSV*]
PTC_codon            54
NMD_predicted        YES
```

Always present: `DS_AG/AL/DG/DL`, `DP_AG/AL/DG/DL`, and the eight
`DS_*_REF`/`DS_*_ALT` probabilities each delta is the difference of.

With `transcripts=` you also get the SAI-10k-calc columns — `Any_splicing_aberration`,
`Exon_skipping`, `Intron_retention`, `Pseudoexon_activation`,
`Partial_exon_deletion`, `Partial_intron_retention`, `Increased_exon_inclusion`,
their sizes and frameshift flags — plus `Consequence` (a one-line summary),
`Event_confidence` / `Confidence_note`, and the protein prediction
(`Protein_consequence`, `PTC_codon`, `NMD_predicted`). About 110 columns in all.

## score() arguments

* `sai10k=False` — skip classification, return scores only
* `raw_scores=False` — drop the REF/ALT columns
* `include_unscored=True` — emit a row with `scored=False` for variants outside
  every transcript, rather than dropping them silently

## One-shot helper

```python
from fastspliceai import score_variants

score_variants(variants, reference=..., annotation=..., models=...)
```

Builds an engine, scores, throws it away. Loading the models takes about a
second, so reuse a `SpliceAI` instance for anything more than a single batch.

For process-level parallelism use the CLI's `-j`. It is rarely worth it: the
five models already run concurrently across all cores inside one process.

---

*Part of the [fast-splice-ai](../README.md) documentation.*
