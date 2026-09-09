#!/usr/bin/env python
"""Score two GRCh38 variants and print the results.

Everything is passed explicitly - nothing is discovered from the environment.

    python examples/grch38_example.py REFERENCE.fa ann.txt gp.txt /path/to/spliceai/models

ann.txt and gp.txt come from a one-off:

    fastspliceai-tx MANE.GRCh38.v1.5.refseq_genomic.gff.gz \
        -o ann.txt --out-genepred gp.txt --name-field transcript --name-prefix RefSeqTx-
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastspliceai import SpliceAI  # noqa: E402

if len(sys.argv) != 5:
    sys.exit(__doc__)
reference, annotation, transcripts, models = sys.argv[1:5]

sai = SpliceAI(reference=reference, annotation=annotation,
               transcripts=transcripts, models=models, distance=500)

df = sai.score([
    "9-128504906-T-G",   # GLE1, the T of the canonical GT donor after exon 1
    "9-21994137-A-C",    # CDKN2A p14ARF donor +2 -- MANE Plus Clinical only
])

cols = ["#CHROM", "POS", "REF", "ALT", "SYMBOL", "DS_AG", "DS_AL", "DS_DG",
        "DS_DL", "DP_DL", "Any_splicing_aberration", "Partial_exon_deletion"]
print(df[[c for c in cols if c in df]].to_string(index=False))
