"""Predicted protein consequence for each splicing event.

The R original has four separate sequence builders (skip / pseudoexon /
retention / partial). They all reduce to the same thing: take the transcript's
exon list, alter it, translate, and compare with wild type. So there is one
builder here and one translator.

Output matches SAI-10k-calc's format - three wild-type residues of context, then
the divergent sequence in brackets, translated to the first stop:

    LGT[QLGRCVRHLW*]

plus two things the R tool does not report: where the premature stop lands, and
whether it is expected to trigger nonsense-mediated decay.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASES = "TCAG"
AAS = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODON = {a + b + c: AAS[i * 16 + j * 4 + k]
         for i, a in enumerate(BASES) for j, b in enumerate(BASES) for k, c in enumerate(BASES)}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# a PTC more than this far upstream of the last exon-exon junction triggers NMD
NMD_RULE_NT = 50


def _rc(s):
    return s.translate(COMP)[::-1]


def translate(dna):
    """Translate to the first stop. Returns (peptide, hit_stop)."""
    out = []
    for i in range(0, len(dna) - 2, 3):
        aa = CODON.get(dna[i:i + 3].upper())
        if aa is None:
            break
        out.append(aa)
        if aa == "*":
            return "".join(out), True
    return "".join(out), False


class _Tx:
    """The bits of a transcript this module needs, taken from the exon table."""
    __slots__ = ("name", "chrom", "strand", "cdsStart", "cdsEnd", "exons")

    def __init__(self, group):
        g = group.sort_values("eNum")
        first = g.iloc[0]
        self.name = first["name"]
        self.chrom = str(first["chrom"])
        self.strand = "+" if int(first["strand"]) == 1 else "-"
        self.cdsStart = int(first["cdsStart"])
        self.cdsEnd = int(first["cdsEnd"])
        # already in transcript order, because eNum is
        self.exons = [(int(r.eStartAdj), int(r.eEnd)) for r in g.itertuples()]


def native_exons(tx):
    """(start, end) 1-based inclusive, in transcript order."""
    return list(tx.exons)


def altered_exons(row, tx, ex):
    """Apply the predicted event to the exon list. None if not modelled."""
    ex = list(ex)
    plus = tx.strand == "+"

    if row.get("Exon_skipping") == "YES" and row.get("Lost_exons"):
        lost = {int(x) for x in str(row["Lost_exons"]).split(",") if str(x).strip().isdigit()}
        keep = [e for n, e in enumerate(ex, start=1) if n not in lost]
        return keep if keep else None

    if row.get("Pseudoexon_activation") == "YES" and pd.notna(row.get("Pseudoexon_start")):
        s, e = int(row["Pseudoexon_start"]), int(row["Pseudoexon_end"])
        out = list(ex) + [(s, e)]
        out.sort(key=lambda p: p[0], reverse=not plus)
        return out

    if row.get("Intron_retention") == "YES" and pd.notna(row.get("Retained_intron")):
        i = int(row["Retained_intron"])                # intron i lies between exon i and i+1
        if 1 <= i < len(ex):
            a, b = ex[i - 1], ex[i]
            merged = (min(a[0], b[0]), max(a[1], b[1]))
            return ex[: i - 1] + [merged] + ex[i + 1:]
        return None

    # partial exon deletion / partial intron retention: one boundary moves
    five, three = row.get("bp_5prime") or 0, row.get("bp_3prime") or 0
    if five or three:
        n = _affected_exon(row, ex)
        if n is None:
            return None
        s, e = ex[n]
        if plus:
            s, e = s + int(five), e + int(three)
        else:
            s, e = s - int(three), e - int(five)
        if s >= e:
            return None
        return ex[:n] + [(s, e)] + ex[n + 1:]
    return None


def _affected_exon(row, ex):
    """Index of the exon the cryptic site sits in, by nearest boundary."""
    pos = int(row["POS"])
    best, dist = None, None
    for i, (s, e) in enumerate(ex):
        d = min(abs(pos - s), abs(pos - e))
        if dist is None or d < dist:
            best, dist = i, d
    return best


def _spliced(fa, chrom, exons, strand, variant=None):
    """Concatenate exons in transcript order, applying the variant if inside."""
    parts, junctions, total = [], [], 0
    for s, e in sorted(exons, key=lambda p: p[0]):
        seq = fa[chrom][s - 1:e].seq.upper()
        if variant is not None:
            pos, ref, alt = variant
            if s <= pos <= e:
                off = pos - s
                if seq[off:off + len(ref)] != ref.upper():
                    return None, None
                seq = seq[:off] + alt.upper() + seq[off + len(ref):]
        parts.append(seq)
    full = "".join(parts)
    if strand == "-":
        full = _rc(full)
        lengths = [len(p) for p in reversed(parts)]
    else:
        lengths = [len(p) for p in parts]
    for n in lengths[:-1]:
        total += n
        junctions.append(total)
    return full, junctions


def _cds_offset(tx, exons, strand):
    """Where the annotated start codon falls in the spliced sequence."""
    first = int(tx.cdsStart) + 1 if strand == "+" else int(tx.cdsEnd)
    ordered = sorted(exons, key=lambda p: p[0], reverse=(strand == "-"))
    off = 0
    for s, e in ordered:
        if s <= first <= e:
            return off + (first - s if strand == "+" else e - first)
        off += e - s + 1
    return None


def predict(df, exons, fasta):
    """Add aaseq / PTC / NMD columns to a classify() result."""
    from pyfaidx import Fasta

    fa = fasta if not isinstance(fasta, (str, bytes)) else Fasta(str(fasta), rebuild=False)
    # only the transcripts these rows touch - building _Tx for the whole table
    # is O(transcripts) per call and dominates everything else
    needed = set(df["name"].dropna().unique()) if "name" in df else set()
    subset = exons[exons["name"].isin(needed)] if needed else exons.iloc[0:0]
    tx_by_name = {n: _Tx(g) for n, g in subset.groupby("name")}
    aaseq, ptc, nmd, note = [], [], [], []

    for _, row in df.iterrows():
        tx = tx_by_name.get(row.get("name"))
        if tx is None or row.get("Any_splicing_aberration") != "YES":
            aaseq.append(None); ptc.append(np.nan); nmd.append(None); note.append(None)
            continue
        ex = native_exons(tx)
        alt_ex = altered_exons(row, tx, ex)
        if alt_ex is None:
            aaseq.append(None); ptc.append(np.nan); nmd.append(None)
            note.append("cannot determine"); continue

        var = (int(row["POS"]), str(row["REF"]), str(row["ALT"]))
        wt_seq, _ = _spliced(fa, tx.chrom, ex, tx.strand)
        mt_seq, junc = _spliced(fa, tx.chrom, alt_ex, tx.strand, var)
        if wt_seq is None or mt_seq is None:
            aaseq.append(None); ptc.append(np.nan); nmd.append(None)
            note.append("reference mismatch"); continue
        wt_off = _cds_offset(tx, ex, tx.strand)
        mt_off = _cds_offset(tx, alt_ex, tx.strand)
        if wt_off is None or mt_off is None:
            aaseq.append(None); ptc.append(np.nan); nmd.append(None)
            note.append("native start or stop is lost"); continue

        wt_aa, _ = translate(wt_seq[wt_off:])
        mt_aa, stopped = translate(mt_seq[mt_off:])
        i = 0
        while i < min(len(wt_aa), len(mt_aa)) and wt_aa[i] == mt_aa[i]:
            i += 1
        if i >= len(mt_aa) and len(wt_aa) == len(mt_aa):
            aaseq.append("does not affect coding region")
            ptc.append(np.nan); nmd.append(None); note.append(None); continue
        aaseq.append("{}[{}]".format(wt_aa[max(0, i - 3):i], mt_aa[i:]))

        if stopped and len(mt_aa) < len(wt_aa):
            stop_nt = mt_off + len(mt_aa) * 3          # nt index of the stop codon end
            last_junction = junc[-1] if junc else None
            ptc.append(len(mt_aa))
            nmd.append("YES" if last_junction is not None
                       and stop_nt < last_junction - NMD_RULE_NT else "NO")
        else:
            ptc.append(np.nan); nmd.append(None)
        note.append(None)

    df = df.copy()
    df["Protein_consequence"] = aaseq
    df["PTC_codon"] = ptc
    df["NMD_predicted"] = nmd
    df["Protein_note"] = note

    # "in-frame" on its own is misleading: an in-frame pseudoexon can carry a
    # stop codon and be a complete null. Say so in the summary.
    if "Consequence" in df.columns:
        def _append(r):
            if pd.isna(r["PTC_codon"]):
                return r["Consequence"]
            tail = "premature stop at codon {:.0f}".format(r["PTC_codon"])
            if r["NMD_predicted"] == "YES":
                tail += ", NMD predicted"
            elif r["NMD_predicted"] == "NO":
                tail += ", escapes NMD"
            return "{} + {}".format(r["Consequence"], tail)
        df["Consequence"] = df.apply(_append, axis=1)
    return df
