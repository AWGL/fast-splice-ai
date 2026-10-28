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
# an alternative outcome scoring below this fraction of the strongest is not a
# plausible competitor and does not affect the loss-of-function call
COMPETITOR_FRACTION = 0.25


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


def exons_for_outcome(outcome, row, tx, ex):
    """Exon list for one *enumerated* outcome, not just the event classify()
    named. Each outcome carries the geometry needed to rebuild it - which exon,
    and the genomic position of the moved boundary or the gained exon - so an
    alternative fate can be translated as readily as the top one.

    Outcomes with no geometry (the partial-exon events, whose boundary shift
    lives in the row's bp_5prime/bp_3prime) fall back to the row.
    """
    ex = list(ex)
    plus = tx.strand == "+"
    event = outcome.get("event")
    n = outcome.get("exon")

    if event == "exon skipping" and n:
        lost = {int(x) for x in str(row.get("Lost_exons") or "").split(",")
                if str(x).strip().isdigit()}
        # the row's Lost_exons may name a run of exons; trust it only when its
        # combined size is the size this outcome is about
        combined = row.get("Lost_exons_combined_size")
        if not lost or pd.isna(combined) or int(combined) != int(outcome["bp"]):
            lost = {int(n)}
        keep = [e for i, e in enumerate(ex, start=1) if i not in lost]
        return keep if keep else None

    if event == "pseudoexon" and outcome.get("start"):
        out = ex + [(int(outcome["start"]), int(outcome["end"]))]
        out.sort(key=lambda p: p[0], reverse=not plus)
        return out

    if event in ("acceptor shift", "donor shift") and outcome.get("new_pos") and n:
        i = int(n) - 1
        if not 0 <= i < len(ex):
            return None
        s, e = ex[i]
        # an acceptor opens the exon and a donor closes it - which end of the
        # genomic interval that is depends on the strand
        if (event == "acceptor shift") == plus:
            s = int(outcome["new_pos"])
        else:
            e = int(outcome["new_pos"])
        if s >= e:
            return None
        return ex[:i] + [(s, e)] + ex[i + 1:]

    return altered_exons(row, tx, ex)


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


def _genomic_at(exons, strand, idx):
    """Genomic coordinate of spliced-sequence position idx (0-based)."""
    for s_, e_ in sorted(exons, key=lambda p: p[0], reverse=(strand == "-")):
        n = e_ - s_ + 1
        if idx < n:
            return (s_ + idx) if strand == "+" else (e_ - idx)
        idx -= n
    return None


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


def _consequence(fa, tx, ex, alt_ex, var, wt):
    """Translate one altered exon list.

    Returns a dict; "disruptive" is True / False / None, None meaning the
    outcome could not be reconstructed. Never raises.
    """
    wt_aa, wt_stop_pos = wt
    if alt_ex is None:
        return {"note": "cannot determine", "disruptive": None}
    mt_seq, junc = _spliced(fa, tx.chrom, alt_ex, tx.strand, var)
    if mt_seq is None:
        return {"note": "reference mismatch", "disruptive": None}
    mt_off = _cds_offset(tx, alt_ex, tx.strand)
    if mt_off is None:
        # the exon carrying the annotated ATG is gone; there is no ORF left to
        # translate, which is a null however you read it
        return {"note": "native start or stop is lost", "disruptive": True}

    mt_aa, stopped = translate(mt_seq[mt_off:])
    i = 0
    while i < min(len(wt_aa), len(mt_aa)) and wt_aa[i] == mt_aa[i]:
        i += 1
    if i >= len(mt_aa) and len(wt_aa) == len(mt_aa):
        return {"aaseq": "does not affect coding region", "disruptive": False}

    out = {"aaseq": "{}[{}]".format(wt_aa[max(0, i - 3):i], mt_aa[i:])}
    if not stopped:
        # ran off the end of the transcript: the native stop is gone
        out["disruptive"] = True
        out["note"] = "stop codon lost"
        return out

    # A stop that is merely *earlier in the peptide* is not a premature stop:
    # any in-frame deletion shortens the protein while still using the native
    # stop codon. The question is whether the codon the ribosome terminates on
    # is the annotated one, so compare where it actually sits in the genome.
    mt_stop_pos = _genomic_at(alt_ex, tx.strand, mt_off + (len(mt_aa) - 1) * 3)
    if wt_stop_pos is not None and mt_stop_pos == wt_stop_pos:
        out["disruptive"] = False                  # in-frame, native stop kept
        return out

    stop_nt = mt_off + len(mt_aa) * 3              # nt index of the stop codon end
    last_junction = junc[-1] if junc else None
    out["ptc"] = len(mt_aa)
    out["nmd"] = ("YES" if last_junction is not None
                  and stop_nt < last_junction - NMD_RULE_NT else "NO")
    out["disruptive"] = True
    return out


def predict(df, exons, fasta):
    """Add aaseq / PTC / NMD columns to a classify() result."""
    from pyfaidx import Fasta

    fa = fasta if not isinstance(fasta, (str, bytes)) else Fasta(str(fasta), rebuild=False)
    # only the transcripts these rows touch - building _Tx for the whole table
    # is O(transcripts) per call and dominates everything else
    needed = set(df["name"].dropna().unique()) if "name" in df else set()
    subset = exons[exons["name"].isin(needed)] if needed else exons.iloc[0:0]
    tx_by_name = {n: _Tx(g) for n, g in subset.groupby("name")}
    wt_cache = {}
    aaseq, ptc, nmd, note, lof, lof_note = [], [], [], [], [], []
    has_outcomes = "Predicted_outcomes" in df.columns

    for _, row in df.iterrows():
        tx = tx_by_name.get(row.get("name"))
        called = row.get("Any_splicing_aberration") == "YES"
        outs = row["Predicted_outcomes"] if has_outcomes else None
        outs = outs if isinstance(outs, list) else []

        if tx is None or not called:
            aaseq.append(None); ptc.append(np.nan); nmd.append(None); note.append(None)
            lof.append(None); lof_note.append(None)
            continue

        ex = native_exons(tx)
        if tx.name not in wt_cache:
            wt_seq, _ = _spliced(fa, tx.chrom, ex, tx.strand)
            wt_off = _cds_offset(tx, ex, tx.strand)
            wt = None
            if wt_seq is not None and wt_off is not None:
                aa, stopped = translate(wt_seq[wt_off:])
                # where the annotated stop codon sits, so an outcome that keeps
                # it can be told apart from one that creates a new one
                wt = (aa, _genomic_at(ex, tx.strand, wt_off + (len(aa) - 1) * 3)
                          if stopped else None)
            wt_cache[tx.name] = wt
        wt = wt_cache[tx.name]
        var = (int(row["POS"]), str(row["REF"]), str(row["ALT"]))
        if wt is None:
            aaseq.append(None); ptc.append(np.nan); nmd.append(None)
            note.append("native start or stop is lost")
            lof.append(None); lof_note.append(None)
            continue

        # the row's own call, for the reported protein consequence
        seen = {}
        def _run(alt_ex):
            key = tuple(alt_ex) if alt_ex is not None else None
            if key not in seen:
                seen[key] = _consequence(fa, tx, ex, alt_ex, var, wt)
            return seen[key]

        main = _run(altered_exons(row, tx, ex))
        aaseq.append(main.get("aaseq"))
        ptc.append(main.get("ptc", np.nan))
        nmd.append(main.get("nmd"))
        note.append(main.get("note"))

        if not outs:
            lof.append(None)
            lof_note.append("no outcome could be enumerated")
            continue

        # Every plausible outcome gets a real translation, not just the top
        # one: an in-frame alternative can still carry a stop codon, and
        # judging it by bp % 3 alone would call that outcome harmless.
        # Only real competitors get a vote - an alternative scoring a tenth of
        # the top one is not a plausible fate, and letting it dilute a
        # confident call would report UNCERTAIN for variants that are not.
        # Every outcome is still listed in Predicted_outcomes.
        top = max(o["score"] for o in outs)
        rivals = [o for o in outs if o["score"] >= COMPETITOR_FRACTION * top]
        n_bad = n_ok = n_unknown = truncating = 0
        for o in rivals:
            res = _run(exons_for_outcome(o, row, tx, ex))
            hit = res.get("disruptive")
            # a frameshift is disruptive whatever the reconstruction said, but
            # an outcome we could not rebuild is unknown, not harmless
            if o["frameshift"] or hit is True:
                n_bad += 1
                if hit is True and not o["frameshift"] and "ptc" in res:
                    truncating += 1
            elif hit is None:
                n_unknown += 1
            else:
                n_ok += 1
        if n_unknown or (n_bad and n_ok):
            lof.append("UNCERTAIN")
        else:
            lof.append("YES" if n_bad else "NO")
        detail = "{}/{} plausible outcomes disruptive".format(n_bad, len(rivals))
        if truncating:
            detail += ", {} of them in-frame but truncating".format(truncating)
        if n_unknown:
            detail += ", {} could not be reconstructed".format(n_unknown)
        ignored = len(outs) - len(rivals)
        if ignored:
            detail += "; {} weaker outcome{} not counted".format(
                ignored, "" if ignored == 1 else "s")
        lof_note.append(detail)

    df = df.copy()
    df["Protein_consequence"] = aaseq
    df["PTC_codon"] = ptc
    df["NMD_predicted"] = nmd
    df["Protein_note"] = note

    # "in-frame" on its own is misleading: an in-frame pseudoexon can carry a
    # stop codon and be a complete null. Say so, and drop the in-frame claim
    # rather than printing "- in-frame + premature stop" in one breath.
    if "Consequence" in df.columns:
        def _append(r):
            if pd.isna(r["PTC_codon"]):
                return r["Consequence"]
            tail = "premature stop at codon {:.0f}".format(r["PTC_codon"])
            if r["NMD_predicted"] == "YES":
                tail += ", NMD predicted"
            elif r["NMD_predicted"] == "NO":
                tail += ", escapes NMD"
            return "{} + {}".format(r["Consequence"].replace(" - in-frame", ""), tail)
        df["Consequence"] = df.apply(_append, axis=1)

    if has_outcomes:
        df["Predicted_LOF"] = lof
        df["LOF_note"] = lof_note
    return df
