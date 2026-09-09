"""Gene annotation table and reference genome access."""
from __future__ import annotations

from pathlib import Path

import numpy as np

BUNDLED = {"grch37": "grch37.txt", "grch38": "grch38.txt"}


def normalise_chrom(source: str, target: str) -> str:
    """Match the 'chr' prefix convention of `target`."""
    s_pre, t_pre = source.startswith("chr"), target.startswith("chr")
    if s_pre and not t_pre:
        return source[3:]
    if t_pre and not s_pre:
        return "chr" + source
    return source


class Annotator:
    """Transcript table plus an indexed FASTA.

    Upstream SpliceAI scans all ~20k transcripts three times per variant. Here
    the transcripts are bucketed by chromosome and sorted by start, so a lookup
    is a binary search over one chromosome instead of a full table scan.
    """

    def __init__(self, ref_fasta, annotations):
        import pandas as pd
        from pyfaidx import Fasta

        if isinstance(annotations, pd.DataFrame):
            df = annotations.copy()
            df["CHROM"] = df["CHROM"].astype(object)
        else:
            key = str(annotations).lower()
            if key in BUNDLED:
                annotations = Path(__file__).with_name("annotations") / BUNDLED[key]
            df = pd.read_csv(annotations, sep="\t", dtype={"CHROM": object})
        missing = {"#NAME", "CHROM", "STRAND", "TX_START", "TX_END", "EXON_START", "EXON_END"} - set(df.columns)
        if missing:
            raise ValueError("annotation file is missing columns: {}".format(sorted(missing)))

        self.genes = df["#NAME"].to_numpy()
        self.chroms = df["CHROM"].to_numpy()
        self.strands = df["STRAND"].to_numpy()
        self.tx_starts = df["TX_START"].to_numpy() + 1
        self.tx_ends = df["TX_END"].to_numpy()
        self.exon_bounds = [
            np.union1d(
                np.asarray([int(i) for i in s.split(",") if i], np.int64) + 1,
                np.asarray([int(i) for i in e.split(",") if i], np.int64),
            )
            for s, e in zip(df["EXON_START"].to_numpy(), df["EXON_END"].to_numpy())
        ]

        # per-chromosome index, sorted by transcript start
        self._index = {}
        for chrom in np.unique(self.chroms):
            idx = np.nonzero(self.chroms == chrom)[0]
            idx = idx[np.argsort(self.tx_starts[idx], kind="stable")]
            self._index[chrom] = (idx, self.tx_starts[idx])

        self.ref_fasta = Fasta(str(ref_fasta), rebuild=False)
        self._fasta_keys = list(self.ref_fasta.keys())

    # -- lookups ---------------------------------------------------------
    def fasta_chrom(self, chrom: str) -> str:
        return normalise_chrom(chrom, self._fasta_keys[0])

    def transcripts(self, chrom: str, pos: int):
        """Indices of transcripts spanning `pos`, in table order."""
        chrom = normalise_chrom(chrom, str(self.chroms[0]))
        entry = self._index.get(chrom)
        if entry is None:
            return np.empty(0, np.int64)
        idx, starts = entry
        cand = idx[: np.searchsorted(starts, pos, side="right")]
        hit = cand[self.tx_ends[cand] >= pos]
        return np.sort(hit)

    def dist_to_boundaries(self, idx: int, pos: int):
        """(distance to tx start, to tx end, to nearest exon boundary)."""
        bounds = self.exon_bounds[idx] - pos
        nearest = bounds[np.argmin(np.abs(bounds))] if bounds.size else 0
        return (
            int(self.tx_starts[idx] - pos),
            int(self.tx_ends[idx] - pos),
            int(nearest),
        )

    def sequence(self, chrom: str, start: int, stop: int) -> str:
        """0-based half-open slice; returns '' if out of range."""
        try:
            return self.ref_fasta[self.fasta_chrom(chrom)][max(start, 0):stop].seq
        except (IndexError, ValueError, KeyError):
            return ""
