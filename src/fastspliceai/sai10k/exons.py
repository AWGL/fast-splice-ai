"""Transcript/exon tables for SAI-10k-calc, built from a genePred table.

Pure pandas. This is the Python equivalent of the transcript preparation in
SAI-10k-calc's spliceAI_parser.R, which uses tidyverse rowwise() joins.

Coordinates follow genePred: eStart is 0-based, eEnd is 1-based. `eStartAdj`
is the 1-based exon start (eStart + 1), matching the R code's naming.
"""
from __future__ import annotations

import pandas as pd

GENEPRED_COLS = [
    "name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd",
    "exonCount", "exonStarts", "exonEnds", "name2", "exonFrames",
]


def load_genepred(path):
    """Read a UCSC genePred-style table (ncbiRefSeq or refGene)."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = set(GENEPRED_COLS) - set(df.columns)
    if missing:
        raise ValueError("transcript table missing columns: {}".format(sorted(missing)))
    for c in ("txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"):
        df[c] = df[c].astype(int)
    return df


def _split_ints(s):
    return [int(x) for x in str(s).split(",") if x not in ("", "nan")]


def build_exon_table(tx):
    """Explode transcripts into one row per exon, with neighbour and intron info.

    eNum counts exons in *transcript* order, so on the minus strand exon 1 is
    the right-most genomic exon. prev_/next_ therefore also follow transcript
    order, matching the lag()/lead() in the R implementation.
    """
    rows = []
    for r in tx.itertuples(index=False):
        starts, ends = _split_ints(r.exonStarts), _split_ints(r.exonEnds)
        frames = _split_ints(r.exonFrames) or [-1] * len(starts)
        strand = 1 if r.strand == "+" else -1
        n = len(starts)
        order = range(n) if strand == 1 else range(n - 1, -1, -1)
        for e_num, i in enumerate(order, start=1):
            rows.append({
                "name": r.name, "Gene": r.name2, "chrom": r.chrom, "strand": strand,
                "exonCount": n, "eNum": e_num,
                "eStart": starts[i], "eEnd": ends[i], "eFrame": frames[i],
                "cdsStart": r.cdsStart, "cdsEnd": r.cdsEnd,
            })
    df = pd.DataFrame(rows).sort_values(["name", "eNum"]).reset_index(drop=True)
    df["eStartAdj"] = df["eStart"] + 1

    g = df.groupby("name", sort=False)
    for col in ("eStart", "eEnd", "eFrame"):
        df["prev_" + col] = g[col].shift(1)
        df["next_" + col] = g[col].shift(-1)
    first, last = df["eNum"] == 1, df["eNum"] == df["exonCount"]
    for col in ("eStart", "eEnd", "eFrame"):
        df.loc[first, "prev_" + col] = pd.NA
        df.loc[last, "next_" + col] = pd.NA

    plus = df["strand"] == 1
    df["intronStart"] = (df["prev_eEnd"] + 1).where(plus, df["eEnd"] + 1)
    df["intronEnd"] = (df["eStart"] - 1).where(plus, df["prev_eStart"] - 1)
    df["intron_num"] = (df["eNum"] - 1).where(~first, pd.NA)
    df.loc[first, ["intronStart", "intronEnd"]] = pd.NA

    df["next_intronStart"] = (df["eEnd"] + 1).where(plus, df["next_eEnd"] + 1)
    df["next_intronEnd"] = (df["next_eStart"] - 1).where(plus, df["eStart"] - 1)
    df["next_intron_num"] = df["eNum"].where(~last, pd.NA)
    df.loc[last, ["next_intronStart", "next_intronEnd"]] = pd.NA

    df["exon_size"] = df["eEnd"] - df["eStart"]
    df["intron_size"] = df["intronEnd"] - df["intronStart"]
    # matches make_consensus_table() in the R original: intronEnd is derived from
    # the 0-based exon start, so the 1-based last intron base is one higher
    df["intronEndAdj"] = df["intronEnd"] + 1
    return df
