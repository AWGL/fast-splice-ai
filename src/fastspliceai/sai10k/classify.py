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

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLICEAI_FIELDS = ["ALLELE", "SYMBOL", "DS_AG", "DS_AL", "DS_DG", "DS_DL",
                   "DP_AG", "DP_AL", "DP_DG", "DP_DL"]
REFALT_FIELDS = ["DS_AG_REF", "DS_AG_ALT", "DS_AL_REF", "DS_AL_ALT",
                 "DS_DG_REF", "DS_DG_ALT", "DS_DL_REF", "DS_DL_ALT"]


@dataclass
class Thresholds:
    """Defaults match SAI-10k-calc's optparse defaults."""
    GEX_size_MIN: float = 25
    GEX_size_MAX: float = 500
    AG_T: float = 0.2
    DG_T: float = 0.2
    AGDG_T: float = 0.0
    DS_GL_T: float = -0.2
    DS_AGDG_MIN_T: float = 0.02
    DS_AGDG_MAX_T: float = 0.2
    DS_ALDL_MIN_T: float = 0.02
    DS_ALDL_MAX_T: float = 0.2


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
    """Join each variant to the exon of its transcript that lies closest."""
    key = "SYMBOL" if variants["SYMBOL"].iloc[0].startswith("RefSeqTx-") else "Gene"
    ex = exons.copy()
    ex["SYMBOL"] = ("RefSeqTx-" + ex["name"]) if key == "SYMBOL" else ex["Gene"]
    ex["chrom_nochr"] = ex["chrom"].str.replace("^chr", "", regex=True)
    v = variants.copy()
    v["chrom_nochr"] = v["#CHROM"].astype(str).str.replace("^chr", "", regex=True)
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
    """Return the SAI-10k-calc prediction table (without the *_aaseq columns)."""
    t = t or Thresholds()
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
    by_tx = {n: g for n, g in exons.groupby("name")}
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

    d["Used_RefSeq_Transcript"] = d["name"]
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
    return "; ".join(out) if out else "No splicing aberration predicted"
