#!/usr/bin/env python
"""Score a set of GRCh38 variants and print scores, splicing consequence and
predicted protein effect.

    python examples/grch38_example.py REFERENCE.fa ann.txt gp.txt /path/to/spliceai/models

ann.txt / gp.txt are the checked-in MANE v1.5 tables under data/mane_v1.5/, or
build your own from a GFF with fastspliceai-tx.

Every coordinate below was derived from published HGVS and checked against the
reference base, so they can be verified rather than taken on trust.
"""
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastspliceai import SpliceAI  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s pid%(process)d] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stderr)

VARIANTS = [
    # --- splice-site variants -------------------------------------------
    ("chr9-128504906-T-G",  "GLE1",   "donor +2, canonical GT"),
    ("chr9-21994137-A-C",   "CDKN2A", "p14ARF donor +2 - MANE Plus Clinical only"),
    ("chr8-140300616-T-G",  "TRPS1",  "exon 11 skipping"),
    # --- deep intronic, known pseudoexon-activating ---------------------
    ("chr17-31161118-T-G",  "NF1",    "c.288+2025T>G   neurofibromatosis"),
    ("chr1-215891198-T-C",  "USH2A",  "c.7595-2144A>G  Usher syndrome 2"),
    ("chr11-108309110-A-G", "ATM",    "c.5763-1050A>G  ataxia-telangiectasia"),
    ("chr3-193618201-G-A",  "OPA1",   "c.610+364G>A    optic atrophy"),
    # --- deep intronic that SpliceAI does NOT flag (known weak spots) ---
    ("chr7-117639961-C-T",  "CFTR",   "c.3718-2477C>T  3849+10kbC>T"),
    ("chr12-88101183-T-C",  "CEP290", "c.2991+1655A>G  LCA10"),
    ("chr17-80104542-T-G",  "GAA",    "c.-32-13T>G     Pompe IVS1"),
]

_t0 = time.perf_counter()
reference, annotation, transcripts, models = sys.argv[1:5]
sai = SpliceAI(reference=reference, annotation=annotation, transcripts=transcripts,
               models=models, distance=500)

df = sai.score([v for v, _, _ in VARIANTS])
label = {v.split("-")[1]: (g, note) for v, g, note in VARIANTS}
df["gene"] = df["POS"].astype(str).map(lambda p: label.get(p, ("", ""))[0])
df["note"] = df["POS"].astype(str).map(lambda p: label.get(p, ("", ""))[1])
df["DS_max"] = df[["DS_AG", "DS_AL", "DS_DG", "DS_DL"]].max(axis=1)

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

print("\n=== SpliceAI delta scores ===")
print(df[["gene", "note", "#CHROM", "POS", "REF", "ALT", "DS_AG", "DS_AL",
          "DS_DG", "DS_DL", "DS_max"]].to_string(index=False))

print("\n=== predicted consequence (SAI-10k-calc + protein) ===")
print(df[["gene", "SYMBOL", "Consequence"]].to_string(index=False))

print("\n{} variants scored in {:.1f} s total (including start-up)"
      .format(len(VARIANTS), time.perf_counter() - _t0))

hits = df[df["Protein_consequence"].notna()]
if len(hits):
    print("\n=== predicted protein ===")
    print(hits[["gene", "Protein_consequence", "PTC_codon", "NMD_predicted"]]
          .to_string(index=False))
