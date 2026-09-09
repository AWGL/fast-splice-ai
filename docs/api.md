# Python API

For embedding in another application. Build the object once - it loads the five
models and the annotation table - then reuse it; reference windows are cached
across calls.

```python
from fastspliceai import SpliceAI

sai = SpliceAI(
    reference="hg19.fa",
    annotation="refseq_ann.txt",      # or "grch37" / "grch38"
    models="/path/to/spliceai/models",
    transcripts="refseq_genepred.txt", # optional: turns on SAI-10k-calc
    distance=500,
)

df = sai.score([
    "13-32889805-G-A",                        # chr-pos-ref-alt
    "13:32890665:G:A",                        # or colon separated
    ("17", 41276133, "C", "T"),               # or a tuple
    {"chrom": "10", "pos": 94077, "ref": "A", "alt": ["C", "G"]},
])
```

One row per (variant, transcript, allele), as a `pandas.DataFrame`: the delta
scores and positions, the REF/ALT probabilities behind them, and - when
`transcripts=` was given - the SAI-10k-calc aberration calls.

```
#CHROM   POS REF ALT  DS_AG  DS_AL  DS_DG  DS_DL  Any_splicing_aberration Exon_skipping Lost_exons
 chr10 94555   C   T   0.01   0.18   0.15   0.62                      YES           YES          3
 chr10 94077   A   C   0.15   0.27   0.00   0.05                      YES            NO       None
```

Useful arguments: `sai10k=False` to skip classification, `raw_scores=False` to
drop the REF/ALT columns, `include_unscored=True` to get a row (with
`scored=False`) for variants outside any transcript rather than silently
dropping them. `score_variants(...)` is a one-shot wrapper for scripts.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
