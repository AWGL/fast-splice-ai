"""SAI-10k-calc splicing-aberration classification, ported to Python.

A faithful port of the classification stage of SAI-10k-calc's spliceAI_parser.R
(Canson, Davidson et al., Bioinformatics 2023; Canson, Parsons et al., Genome
Research 2025), MIT licensed.

The R original applies 22 separate `rowwise()` passes over a tibble, which is
row-at-a-time interpretation. Everything here is vectorised over the whole
table; per-transcript lookups are done against pre-grouped dicts and only for
the handful of rows whose predicates actually fire.

Not yet ported: the predicted amino-acid sequences (*_aaseq columns), which
need the reference genome and translation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLICEAI_FIELDS = ["ALLELE", "SYMBOL", "DS_AG", "DS_AL", "DS_DG", "DS_DL",
                   "DP_AG", "DP_AL", "DP_DG", "DP_DL"]
REFALT_FIELDS = ["DS_AG_REF", "DS_AG_ALT", "DS_AL_REF", "DS_AL_ALT",
                 "DS_DG_REF", "DS_DG_ALT", "DS_DL_REF", "DS_DL_ALT"]


# a runner-up this close to the winner makes the predicted position ambiguous
RIVAL_AMBIGUOUS = 0.8
# a gained acceptor/donor pair is only a plausible exon within this size range
GEX_MIN, GEX_MAX = 25, 500


def _genomic(r, offset):
    """Delta positions are transcript-oriented; convert to a genomic coordinate."""
    return int(r["POS"]) + (-offset if r["strand"] == -1 else offset)


def _outcomes(r):
    """Each plausible outcome, and whether it shifts the reading frame.

    A splice-disrupting variant usually has several possible fates - the exon
    is skipped, or one of two cryptic sites is used. Reporting only the
    strongest hides that. But if *every* fate frameshifts, which one occurs
    stops mattering: the protein is lost either way, and that is the call to
    surface.

    Shifts are measured against the native site the variant destroys, using the
    same offset convention as DP_AG/DP_AL/DP_DG/DP_DL.
    """
    outs = []
    # 1. the event classify() actually called, so the two can never disagree
    primary = [
        ("pseudoexon", r["Pseudoexon_activation"], r.get("Gained_exon_size")),
        ("exon skipping", r["Exon_skipping"], r.get("Lost_exons_combined_size")),
        ("intron retention", r["Intron_retention"], r.get("Retained_intron_size")),
    ]
    for event, fired, size in primary:
        if fired == "YES" and pd.notna(size) and size:
            n = int(size)
            o = {"event": event, "bp": n, "frameshift": n % 3 != 0,
                 "score": float(r["DS_AGDG_MAX"] if event == "pseudoexon"
                                else r["DS_ALDL_MAX"]),
                 "exon": int(r["eNum"])}
            if event == "pseudoexon" and pd.notna(r.get("Pseudoexon_start")):
                o["start"], o["end"] = int(r["Pseudoexon_start"]), int(r["Pseudoexon_end"])
            outs.append(o)
    for event, fired in (("acceptor shift", r["Partial_exon_deletion"]),
                         ("acceptor shift", r["Partial_intron_retention"])):
        if fired == "YES":
            n = int(max(abs(r.get("bp_5prime") or 0), abs(r.get("bp_3prime") or 0)))
            if n:
                outs.append({"event": event, "bp": n, "frameshift": n % 3 != 0,
                             "score": float(max(r["DS_AG"], r["DS_DG"]))})

    # 2. alternatives the variant could plausibly produce instead
    cands = r.get("CANDIDATES")
    if not isinstance(cands, list) or not cands:
        return _dedupe(outs)
    lost = {c["kind"].split("_")[0]: c["offset"]
            for c in cands if c["kind"].endswith("_loss")}
    size = r.get("exon_size")
    for c in cands:
        which, event = c["kind"].split("_")
        if event == "loss":
            if pd.isna(size):
                continue
            n = int(size)
            outs.append({"event": "exon skipping", "bp": n, "frameshift": n % 3 != 0,
                         "score": c["score"], "exon": int(r["eNum"])})
        elif which in lost:
            n = abs(c["offset"] - lost[which])
            if n == 0:
                continue
            outs.append({"event": "{} shift".format(which), "bp": n,
                         "frameshift": n % 3 != 0, "score": c["score"],
                         "exon": int(r["eNum"]), "site": which,
                         "new_pos": _genomic(r, c["offset"])})
    # A pseudoexon destroys nothing, so there is no loss to measure shifts
    # against: the outcome is the exon the gained acceptor and donor bracket.
    if not lost:
        acc = [c for c in cands if c["kind"] == "acceptor_gain"]
        don = [c for c in cands if c["kind"] == "donor_gain"]
        plus = r.get("strand", 1) == 1
        for a in acc:
            for dn in don:
                right_way = (a["offset"] < dn["offset"]) if plus else (a["offset"] > dn["offset"])
                if not right_way:
                    continue
                n = abs(a["offset"] - dn["offset"]) + 1
                if not (GEX_MIN <= n <= GEX_MAX):
                    continue
                ga, gd = _genomic(r, a["offset"]), _genomic(r, dn["offset"])
                outs.append({"event": "pseudoexon", "bp": n,
                             "frameshift": n % 3 != 0,
                             "score": round(min(a["score"], dn["score"]), 3),
                             "exon": int(r["eNum"]),
                             "start": min(ga, gd), "end": max(ga, gd)})

    return _dedupe(outs)


def _dedupe(outs):
    """One row per distinct consequence, strongest first."""
    seen, uniq = set(), []
    for o in sorted(outs, key=lambda o: -o["score"]):
        key = (o["event"], o["bp"])
        if key not in seen:
            seen.add(key)
            uniq.append(o)
    return uniq


def rivals_for(r):
    """Which of the four events actually drove this call.

    Only those matter: a tie among acceptor-gain positions says nothing about
    an exon-skipping call driven by the losses.
    """
    events = []
    if r["Pseudoexon_activation"] == "YES" or r["Increased_exon_inclusion"] == "YES":
        events += ["AG", "DG"]
    if r["Exon_skipping"] == "YES" or r["Intron_retention"] == "YES":
        events += ["AL", "DL"]
    if r["Partial_exon_deletion"] == "YES" or r["Partial_intron_retention"] == "YES":
        events += ["AG"] if r["Cryptic_Acceptor_activation"] == "YES" else []
        events += ["DG"] if r["Cryptic_Donor_activation"] == "YES" else []
    return events


@dataclass
class Thresholds:
    """Defaults match SAI-10k-calc's optparse defaults exactly.

    ONE EXCEPTION: DS_ALDL_MIN_T defaults to 0.05 here, where SAI-10k-calc uses
    0.02. At 0.02 an acceptor-loss score of 0.05 - noise - is enough to help
    support an intron-retention call, which produced a visibly spurious call on
    CDKN2A. Pass Thresholds(DS_ALDL_MIN_T=0.02) to reproduce the R tool exactly.

    Override individually, or take a tighter set wholesale:

        SpliceAI(..., thresholds={"DS_ALDL_MIN_T": 0.1})
        SpliceAI(..., thresholds=Thresholds.strict())
    """

    R_DEFAULT_DS_ALDL_MIN_T = 0.02
    GEX_size_MIN: float = 25
    GEX_size_MAX: float = 500
    AG_T: float = 0.2
    DG_T: float = 0.2
    AGDG_T: float = 0.0
    DS_GL_T: float = -0.2
    DS_AGDG_MIN_T: float = 0.02
    DS_AGDG_MAX_T: float = 0.2
    DS_ALDL_MIN_T: float = 0.05          # SAI-10k-calc uses 0.02; see note below
    DS_ALDL_MAX_T: float = 0.2

    @classmethod
    def strict(cls, **overrides):
        """Tighter floors: every supporting score must be a real signal, not noise."""
        base = dict(DS_AGDG_MIN_T=0.1, DS_ALDL_MIN_T=0.1, AGDG_T=0.1)
        base.update(overrides)
        return cls(**base)

    @classmethod
    def make(cls, value):
        """Accept a Thresholds, a dict of overrides, or None."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown thresholds: {}".format(sorted(unknown)))
        return cls(**value)


def parse_spliceai_vcf(path):
    """One row per (record, SpliceAI annotation). Handles the REF/ALT format."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            tag = None
            for kv in f[7].split(";"):
                if kv.startswith("SpliceAI="):
                    tag = kv[len("SpliceAI="):]
            if not tag:
                continue
            for ann in tag.split(","):
                parts = ann.split("|")
                names = SPLICEAI_FIELDS + (REFALT_FIELDS if len(parts) >= 18 else [])
                if len(parts) < len(names):
                    continue
                rec = dict(zip(names, parts))
                rec.update({"#CHROM": f[0], "POS": int(f[1]), "ID": f[2],
                            "REF": f[3], "ALT": f[4], "QUAL": f[5], "FILTER": f[6]})
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in SPLICEAI_FIELDS[2:] + [c for c in REFALT_FIELDS if c in df.columns]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _closest_exon(variants, exons):
    """Join each variant to the exon of its transcript that lies closest.

    The join keys are precomputed by build_exon_table(), and the exon table is
    narrowed to the transcripts these variants actually touch before merging, so
    the cost tracks the number of variants rather than the size of the table.
    Nothing is cached on the frame: pandas deep-copies .attrs on every derived
    operation, so a large object parked there is copied once per groupby group.
    """
    col = ("SYMBOL_refseq" if variants["SYMBOL"].iloc[0].startswith("RefSeqTx-")
           else "SYMBOL_gene")
    v = variants.copy()
    v["chrom_nochr"] = v["#CHROM"].astype(str).str.replace("^chr", "", regex=True)

    wanted = set(v["chrom_nochr"] + "\x00" + v["SYMBOL"])
    keys = exons["chrom_nochr"] + "\x00" + exons[col]
    ex = exons[keys.isin(wanted)].copy()
    ex["SYMBOL"] = ex[col]

    m = v.merge(ex, left_on=["chrom_nochr", "SYMBOL"], right_on=["chrom_nochr", "SYMBOL"],
                how="left", suffixes=("", "_ex"))
    m = m[m["name"].notna()].copy()
    m["dist_exon_start"] = m["POS"] - m["eStart"] - 1
    m["dist_exon_end"] = m["POS"] - m["eEnd"]
    m["exonic"] = np.where((m["dist_exon_start"] > 0) & (m["dist_exon_end"] < 0), "yes", "no")
    m["dist_exon_closest_abs"] = np.minimum(m["dist_exon_start"].abs(), m["dist_exon_end"].abs())
    best = m.groupby(["ID", "SYMBOL"])["dist_exon_closest_abs"].transform("min")
    m = m[m["dist_exon_closest_abs"] == best].copy()
    m["d_from_exon"] = np.select(
        [(m["dist_exon_start"] <= 0) & (m["dist_exon_end"] < 0),
         (m["dist_exon_start"] > 0) & (m["dist_exon_end"] >= 0)],
        [m["dist_exon_start"].abs(), m["dist_exon_end"]], default=0)
    m["d_250bp"] = np.where(m["d_from_exon"] > 250, "YES", "NO")
    m["d_50bp"] = np.where(m["d_from_exon"] > 50, "YES", "NO")
    return m


def _find_pseudoexon_intron(row, by_tx):
    t = by_tx[row["name"]]
    if row["strand"] == 1:
        hit = t[(t["intronStart"] + 50 < row["GEO_AG"]) & (t["intronEndAdj"] - 50 > row["GEO_DG"])]
    else:
        hit = t[(t["intronStart"] + 50 < row["GEO_DG"]) & (t["intronEndAdj"] - 50 > row["GEO_AG"])]
    hit = hit[hit["intron_num"].notna() & hit["prev_eFrame"].notna()]
    if len(hit) != 1:
        return (None, None)
    return (hit["intron_num"].iloc[0], hit["prev_eFrame"].iloc[0])


def _find_retained_intron(row, by_tx):
    t = by_tx[row["name"]]
    if row["strand"] == 1:
        hit = t[(t["intronStart"] == row["GEO_DL"] + 1) & (t["intronEndAdj"] == row["GEO_AL"] - 1)]
    else:
        hit = t[(t["intronStart"] == row["GEO_AL"] + 1) & (t["intronEndAdj"] == row["GEO_DL"] - 1)]
    hit = hit[hit["intron_num"].notna() & hit["prev_eFrame"].notna()]
    if hit.empty:
        return (None, None)
    return (hit["intron_num"].iloc[0], hit["prev_eFrame"].iloc[0])


def _find_exons_lost_gain(row, by_tx, dl_dg, al_ag):
    t = by_tx[row["name"]]
    a, b = row[dl_dg], row[al_ag]
    for x, y in ((a, b), (b, a)):
        if (t["eStartAdj"] == x).any() and (t["eEnd"] == y).any():
            hit = t[(t["eStartAdj"] == x) | (t["eEnd"] == y)]
            nums = ",".join(str(int(n)) for n in dict.fromkeys(hit["eNum"]))
            return (nums, int(hit["exon_size"].sum()))
    return (None, None)


def classify(variants, exons, t=None):
    """Return the SAI-10k-calc prediction table."""
    t = Thresholds.make(t)
    d = _closest_exon(variants, exons)
    if d.empty:
        return d
    include = all(c in d.columns for c in REFALT_FIELDS)

    for s in ("AG", "AL", "DG", "DL"):
        d["GEO_" + s] = d["POS"] + d["DP_" + s]

    # --- gained-exon (GEX) prediction -------------------------------------
    d["DS_AGDG_MIN"] = d[["DS_AG", "DS_DG"]].min(axis=1)
    d["DS_AGDG_MAX"] = d[["DS_AG", "DS_DG"]].max(axis=1)
    d["DS_AGDG_CUTOFF"] = np.where(
        (d["DS_AGDG_MIN"] >= t.DS_AGDG_MIN_T) & (d["DS_AGDG_MAX"] >= t.DS_AGDG_MAX_T), "PASS", "FAIL")
    d["SS_AGDG_orientation"] = np.where(
        d["strand"] == 1, d["DP_AG"] < d["DP_DG"], d["DP_AG"] > d["DP_DG"])
    d["GEX_size"] = np.where(
        (d["DS_AGDG_CUTOFF"] == "PASS") & d["SS_AGDG_orientation"],
        (d["DP_AG"] - d["DP_DG"]).abs() + 1, np.nan)
    d["GEX_predict"] = np.where(
        (d["GEX_size"] >= t.GEX_size_MIN) & (d["GEX_size"] <= t.GEX_size_MAX), "PASS", "FAIL")

    # --- lost-exon (LEX) and retained-intron (RET) prediction --------------
    d["DS_ALDL_MIN"] = d[["DS_AL", "DS_DL"]].min(axis=1)
    d["DS_ALDL_MAX"] = d[["DS_AL", "DS_DL"]].max(axis=1)
    d["DS_ALDL_CUTOFF"] = np.where(
        (d["DS_ALDL_MIN"] >= t.DS_ALDL_MIN_T) & (d["DS_ALDL_MAX"] >= t.DS_ALDL_MAX_T), "PASS", "FAIL")
    d["SS_ALDL_orientation"] = np.where(
        d["strand"] == 1, d["DP_AL"] < d["DP_DL"], d["DP_AL"] > d["DP_DL"])
    ok = d["DS_ALDL_CUTOFF"] == "PASS"
    d["LEX_predict"] = np.where(ok & d["SS_ALDL_orientation"], (d["DP_AL"] - d["DP_DL"]).abs() + 1, np.nan)
    d["RET_predict"] = np.where(ok & ~d["SS_ALDL_orientation"], (d["DP_AL"] - d["DP_DL"]).abs() - 1, np.nan)

    # --- de novo cryptic acceptor / donor activation ----------------------
    if include:
        d["DG_native_match"] = np.where(
            ((d["GEO_DG"] == d["eEnd"]) & (d["strand"] == 1)) |
            ((d["GEO_DG"] == d["eStart"] + 1) & (d["strand"] == -1)), "PASS", "FAIL")
        d["DG_acceptor_match"] = np.where(
            ((d["GEO_AG"] == d["eStart"] + 1) & (d["strand"] == 1)) |
            ((d["GEO_AG"] == d["eEnd"]) & (d["strand"] == -1)), "PASS", "FAIL")
        acc_diff = d["DS_AG_ALT"] - d["DS_AL_ALT"]
        don_diff = d["DS_DG_ALT"] - d["DS_DL_ALT"]
        a1 = (d["DS_AG"] < 0.2) & (d["DS_AL"] > 0.1) & (d["DS_AG_ALT"] >= 0.9) & (acc_diff >= 0.2)
        a2 = (d["DS_AG"] >= t.AG_T) & (acc_diff > t.DS_GL_T)
        n1 = (d["DS_DG"] >= t.DG_T) & (don_diff > t.DS_GL_T)
        n2 = (d["DS_DG"] < 0.2) & (d["DS_DL"] > 0.1) & (d["DS_DG_ALT"] >= 0.9) & (don_diff >= 0.2)
        big = d["DS_AGDG_MAX"] > t.AGDG_T
        native, acceptor = d["DG_native_match"] == "PASS", d["DG_acceptor_match"] == "PASS"
        d["Cryptic_Acceptor_activation"] = np.where(
            (big & native & ~acceptor & (a1 | a2)) |
            (big & ~native & ~acceptor & (d["DS_AG"] > d["DS_DG"]) & (a1 | a2)), "YES", "NO")
        d["Cryptic_Donor_activation"] = np.where(
            (big & ~native & acceptor & (n1 | n2)) |
            (big & ~native & ~acceptor & (d["DS_AG"] < d["DS_DG"]) & (n1 | n2)), "YES", "NO")
    else:
        d["Cryptic_Acceptor_activation"] = np.where(
            (d["DS_AG"] >= t.AG_T) & (d["DS_AG"] > d["DS_DG"]), "YES", "NO")
        d["Cryptic_Donor_activation"] = np.where(
            (d["DS_DG"] >= t.DG_T) & (d["DS_DG"] > d["DS_AG"]), "YES", "NO")

    # --- orientation and partial sizes ------------------------------------
    d["Cryptic_Acceptor_orientation"] = np.where(
        ((d["strand"] == 1) & (d["prev_eEnd"] - d["GEO_AG"] < 0)) |
        ((d["strand"] == -1) & (d["GEO_AG"] - (d["prev_eStart"] + 1) < 0)), "PASS", "FAIL")
    d["Cryptic_Donor_orientation"] = np.where(
        ((d["strand"] == 1) & (d["GEO_DG"] - (d["next_eStart"] + 1) < 0)) |
        ((d["strand"] == -1) & (d["next_eEnd"] - d["GEO_DG"] < 0)), "PASS", "FAIL")

    d["Gained_exon_size"] = d["GEX_size"]
    d["Lost_exon_size"] = d["LEX_predict"]
    d["Retained_intron_size"] = d["RET_predict"]
    ca = (d["Cryptic_Acceptor_activation"] == "YES") & (d["Cryptic_Acceptor_orientation"] == "PASS")
    cd = (d["Cryptic_Donor_activation"] == "YES") & (d["Cryptic_Donor_orientation"] == "PASS")
    d["bp_5prime"] = np.select(
        [ca & (d["strand"] == 1), ca & (d["strand"] == -1)],
        [d["GEO_AG"] - (d["eStart"] + 1), d["eEnd"] - d["GEO_AG"]], default=0)
    d["bp_3prime"] = np.select(
        [cd & (d["strand"] == 1), cd & (d["strand"] == -1)],
        [d["GEO_DG"] - d["eEnd"], (d["eStart"] + 1) - d["GEO_DG"]], default=0)

    # --- per-transcript lookups (only where the predicate fires) ----------
    needed = set(d["name"].unique())
    by_tx = {n: g for n, g in exons[exons["name"].isin(needed)].groupby("name")}
    need_pe = (d["d_50bp"] == "YES") & (d["GEX_predict"] == "PASS") & d["Gained_exon_size"].notna()
    pe = d[need_pe].apply(lambda r: _find_pseudoexon_intron(r, by_tx), axis=1) if need_pe.any() else pd.Series(dtype=object)
    d["Pseudoexon_intron"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    d["Pseudoexon_intron_type"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    if len(pe):
        d.loc[need_pe, "Pseudoexon_intron"] = [x[0] for x in pe]
        d.loc[need_pe, "Pseudoexon_intron_type"] = [x[1] for x in pe]

    # --- event types ------------------------------------------------------
    d["Partial_intron_retention"] = np.where(
        (d["d_250bp"] == "NO") &
        ((d["Cryptic_Acceptor_activation"] == "YES") | (d["Cryptic_Donor_activation"] == "YES")) &
        (((d["bp_5prime"] < 0) & (d["bp_5prime"] > -251)) |
         ((d["bp_3prime"] > 0) & (d["bp_3prime"] < 251))), "YES", "NO")
    d["Pseudoexon_activation"] = np.where(
        (d["d_50bp"] == "YES") & (d["GEX_predict"] == "PASS") &
        d["Gained_exon_size"].notna() & d["Pseudoexon_intron"].notna(), "YES", "NO")
    d["Partial_exon_deletion"] = np.where(
        (d["d_50bp"] == "NO") & ((d["bp_5prime"] > 0) | (d["bp_3prime"] < 0)), "YES", "NO")
    d["Exon_skipping"] = np.where(
        (d["d_50bp"] == "YES") | d["Lost_exon_size"].isna(), "NO", "YES")
    d["Intron_retention"] = np.where(
        (d["d_50bp"] == "YES") | d["Retained_intron_size"].isna(), "NO", "YES")
    d["Increased_exon_inclusion"] = np.where(
        d["SS_AGDG_orientation"] & (d["DS_AGDG_CUTOFF"] == "PASS") &
        (d["Gained_exon_size"] == d["exon_size"]) &
        (((d["GEO_AG"] == d["eEnd"]) & (d["GEO_DG"] == d["eStart"] + 1)) |
         ((d["GEO_DG"] == d["eEnd"]) & (d["GEO_AG"] == d["eStart"] + 1))), "YES", "NO")

    d["Any_splicing_aberration"] = np.where(
        (d[["Partial_intron_retention", "Pseudoexon_activation", "Partial_exon_deletion",
            "Exon_skipping", "Intron_retention", "Increased_exon_inclusion"]] == "YES").any(axis=1),
        "YES", "NO")

    # --- extra detail per event ------------------------------------------
    pa = d["Pseudoexon_activation"] == "YES"
    d["Pseudoexon_start"] = np.where(pa & d["Pseudoexon_intron"].notna(),
                                     d[["GEO_AG", "GEO_DG"]].min(axis=1), np.nan)
    d["Pseudoexon_end"] = np.where(pa & d["Pseudoexon_intron"].notna(),
                                   d[["GEO_AG", "GEO_DG"]].max(axis=1), np.nan)
    d["Pseudoexon_frameshift"] = np.where(
        pa & d["Pseudoexon_intron_type"].notna() & (d["Pseudoexon_intron_type"] != -1),
        np.where(((d["Pseudoexon_start"] - d["Pseudoexon_end"]).abs() + 1) % 3 != 0, "YES", "NO"), None)

    ir = d["Intron_retention"] == "YES"
    d["Retained_intron"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    d["Retained_intron_type"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    if ir.any():
        res = d[ir].apply(lambda r: _find_retained_intron(r, by_tx), axis=1)
        d.loc[ir, "Retained_intron"] = [x[0] for x in res]
        d.loc[ir, "Retained_intron_type"] = [x[1] for x in res]
    d["Intron_retention_frameshift"] = np.where(
        ir & d["Retained_intron_type"].notna() & (d["Retained_intron_type"] != -1),
        np.where(((d["GEO_AL"] - d["GEO_DL"]).abs() - 1) % 3 != 0, "YES", "NO"), None)

    es = d["Exon_skipping"] == "YES"
    d["Lost_exons"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    d["Lost_exons_combined_size"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    if es.any():
        res = d[es].apply(lambda r: _find_exons_lost_gain(r, by_tx, "GEO_DL", "GEO_AL"), axis=1)
        d.loc[es, "Lost_exons"] = [x[0] for x in res]
        d.loc[es, "Lost_exons_combined_size"] = [x[1] for x in res]
    d["Exon_skipping_frameshift"] = np.where(
        es & d["Lost_exons_combined_size"].notna(),
        np.where(d["Lost_exons_combined_size"] % 3 != 0, "YES", "NO"), None)

    ie = d["Increased_exon_inclusion"] == "YES"
    d["Exon_with_increased_inclusion"] = pd.Series([None] * len(d), index=d.index, dtype=object)
    if ie.any():
        res = d[ie].apply(lambda r: _find_exons_lost_gain(r, by_tx, "GEO_DG", "GEO_AG"), axis=1)
        d.loc[ie, "Exon_with_increased_inclusion"] = [x[0] for x in res]

    # Where the variant sits, VEP-style: "7/11" for exonic, or the intron it
    # falls in. intron_num is the intron 5' of this exon in transcript order,
    # next_intron_num the one 3' of it; both are already strand-aware.
    in_prev = (d["POS"] >= d["intronStart"]) & (d["POS"] <= d["intronEnd"])
    in_next = (d["POS"] >= d["next_intronStart"]) & (d["POS"] <= d["next_intronEnd"])
    n_introns = d["exonCount"] - 1
    d["EXON"] = np.where(d["exonic"] == "yes",
                         d["eNum"].astype(str) + "/" + d["exonCount"].astype(str), None)
    d["INTRON"] = np.select(
        [in_prev, in_next],
        [d["intron_num"].astype("Int64").astype(str) + "/" + n_introns.astype(str),
         d["next_intron_num"].astype("Int64").astype(str) + "/" + n_introns.astype(str)],
        default=None)

    # --- is the outcome disruptive whichever way it goes? -----------------
    outcomes, lof, summary = [], [], []
    for _, r in d.iterrows():
        outs = _outcomes(r)
        outcomes.append(outs)
        if not outs:
            lof.append(None)
            summary.append("no outcome could be enumerated"
                           if r["Any_splicing_aberration"] == "YES" else None)
            continue
        bad = sum(1 for o in outs if o["frameshift"])
        lof.append("YES" if bad == len(outs) else "UNCERTAIN" if bad else "NO")
        summary.append("{}/{} outcomes frameshift".format(bad, len(outs)))
    d["Predicted_outcomes"] = outcomes
    d["Predicted_LOF"] = lof
    d["LOF_note"] = summary

    d["Used_RefSeq_Transcript"] = d["name"]

    # How well supported is each call? The limiting delta score, plus whether
    # the event could be tied to a named intron or exon. An intron-retention
    # call that cannot name an intron is unanchored and should be treated with
    # suspicion, however high the donor-loss score.
    conf, why = [], []
    for _, r in d.iterrows():
        limits, notes = [], []
        if r["Intron_retention"] == "YES" or r["Exon_skipping"] == "YES":
            limits.append(float(r["DS_ALDL_MIN"]))
            if r["Intron_retention"] == "YES" and pd.isna(r["Retained_intron"]):
                notes.append("intron not identified")
            if r["Exon_skipping"] == "YES" and not r["Lost_exons"]:
                notes.append("exons not identified")
        if r["Pseudoexon_activation"] == "YES" or r["Increased_exon_inclusion"] == "YES":
            limits.append(float(r["DS_AGDG_MIN"]))
        if r["Partial_exon_deletion"] == "YES" or r["Partial_intron_retention"] == "YES":
            limits.append(float(max(r["DS_AG"], r["DS_DG"])))
        # A rival site close behind the winner means the predicted *position* is
        # a coin flip even when the score is strong, so the event may be real
        # while the reported coordinates are not.
        for ev in rivals_for(r):
            frac = r.get("RIVAL_" + ev)
            if frac is not None and frac >= RIVAL_AMBIGUOUS:
                notes.append("rival {} site at {:.0%} of the top".format(ev, frac))

        if not limits:
            conf.append(None); why.append(None); continue
        lim = min(limits)
        label = "high" if lim >= 0.5 else "moderate" if lim >= 0.2 else "low"
        if notes:
            label = "low"
        notes.insert(0, "limiting score {:.2f}".format(lim))
        conf.append(label); why.append("; ".join(notes))
    d["Event_confidence"] = conf
    d["Confidence_note"] = why

    d["Consequence"] = d.apply(_consequence, axis=1)
    return d


def _plural(word, items):
    return "{} {}".format(word + ("s" if "," in str(items) else ""), items)


def _frame(flag):
    if flag == "YES":
        return " - frameshift"
    if flag == "NO":
        return " - in-frame"
    return ""


def _consequence(r):
    """A one-line human-readable summary, in the style of SpliceAI-lookup.

    Everything here is assembled from columns computed above; nothing new is
    predicted. Several events can fire for one variant, so they are joined
    with '; '.
    """
    out = []
    if r.get("Exon_skipping") == "YES":
        size = r.get("Lost_exons_combined_size")
        size = " ({:.0f}bp)".format(size) if pd.notna(size) else ""
        which = (_plural("Exon", r["Lost_exons"]) if r.get("Lost_exons") else "Exon")
        out.append("{} skipping{}{}".format(
            which, size, _frame(r.get("Exon_skipping_frameshift"))))
    if r.get("Intron_retention") == "YES":
        size = r.get("Retained_intron_size")
        size = " ({:.0f}bp)".format(size) if pd.notna(size) else ""
        # the intron is only named when its boundaries match an annotated intron
        which = (" {:.0f}".format(float(r["Retained_intron"]))
                 if pd.notna(r.get("Retained_intron")) else "")
        out.append("Intron{} retention{}{}".format(
            which, size, _frame(r.get("Intron_retention_frameshift"))))
    if r.get("Pseudoexon_activation") == "YES":
        size = r.get("Gained_exon_size")
        size = " ({:.0f}bp)".format(size) if pd.notna(size) else ""
        where = (" in intron {:.0f}".format(float(r["Pseudoexon_intron"]))
                 if pd.notna(r.get("Pseudoexon_intron")) else "")
        out.append("Pseudoexon activation{}{}{}".format(
            where, size, _frame(r.get("Pseudoexon_frameshift"))))
    if r.get("Partial_exon_deletion") == "YES":
        n = max(abs(r.get("bp_5prime") or 0), abs(r.get("bp_3prime") or 0))
        out.append("Partial exon deletion" + (" ({:.0f}bp)".format(n) if n else ""))
    if r.get("Partial_intron_retention") == "YES":
        n = max(abs(r.get("bp_5prime") or 0), abs(r.get("bp_3prime") or 0))
        out.append("Partial intron retention" + (" ({:.0f}bp)".format(n) if n else ""))
    if r.get("Increased_exon_inclusion") == "YES":
        e = r.get("Exon_with_increased_inclusion")
        out.append("Increased inclusion of {}".format(_plural("exon", e)) if e
                   else "Increased exon inclusion")
    if not out:
        return "No splicing aberration predicted"
    text = "; ".join(out)
    if r.get("Event_confidence") == "low":
        text += " [low confidence: {}]".format(r.get("Confidence_note") or "")
    return text
