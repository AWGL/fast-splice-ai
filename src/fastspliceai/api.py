"""High-level API for embedding FastSpliceAI in another application.

    from fastspliceai import SpliceAI

    sai = SpliceAI(reference="hg19.fa", annotation="grch37",
                   models="/path/to/spliceai/models",
                   transcripts="ncbiRefSeq.txt",   # optional: enables SAI-10k-calc
                   distance=500)

    df = sai.score(["13-32889805-G-A", ("17", 41276133, "C", "T")])

Construct `SpliceAI` once and reuse it: the five models and the annotation table
are loaded in __init__, and reference windows are cached across calls.

Everything is offline. Nothing here opens a network connection.
"""
from __future__ import annotations

import logging
import re
import time
from collections import namedtuple
from pathlib import Path

import pandas as pd

from .annotator import Annotator
from .engine import Ensemble
from .score import _RefCache, get_delta_scores

log = logging.getLogger("fastspliceai")

Variant = namedtuple("Variant", "chrom pos ref alts")

DELTA_COLS = ["DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL"]
RAW_COLS = ["DS_AG_REF", "DS_AG_ALT", "DS_AL_REF", "DS_AL_ALT",
            "DS_DG_REF", "DS_DG_ALT", "DS_DL_REF", "DS_DL_ALT"]
_VARIANT_RE = re.compile(r"^([^\s:\-]+)[:\-](\d+)[:\-]([A-Za-z*.<>]+)[:\->]+([A-Za-z*.<>,]+)$")


def parse_variant(v):
    """Accept 'chr-pos-ref-alt', 'chr:pos:ref:alt', a 4-tuple, or a mapping."""
    if isinstance(v, Variant):
        return v
    if isinstance(v, str):
        m = _VARIANT_RE.match(v.strip())
        if not m:
            raise ValueError("cannot parse variant {!r}".format(v))
        chrom, pos, ref, alt = m.groups()
        return Variant(chrom, int(pos), ref.upper(), [a.upper() for a in alt.split(",")])
    if isinstance(v, dict):
        alt = v.get("alt", v.get("alts"))
        alts = [alt] if isinstance(alt, str) else list(alt)
        return Variant(str(v["chrom"]), int(v["pos"]), str(v["ref"]).upper(),
                       [str(a).upper() for a in alts])
    if isinstance(v, (tuple, list)) and len(v) == 4:
        chrom, pos, ref, alt = v
        alts = [alt] if isinstance(alt, str) else list(alt)
        return Variant(str(chrom), int(pos), str(ref).upper(), [str(a).upper() for a in alts])
    raise TypeError("unsupported variant {!r}".format(v))


def _read_annotation(path):
    """Parse a SpliceAI annotation table without opening a reference genome."""
    from .annotator import BUNDLED

    key = str(path).lower()
    if key in BUNDLED:
        path = Path(__file__).with_name("annotations") / BUNDLED[key]
    return pd.read_csv(path, sep="\t", dtype={"CHROM": object})


def _is_single_variant(v):
    """True for one variant, False for a collection of them.

    The ambiguous case is a 4-element sequence: ("chr1", 100, "A", "T") is one
    variant, but ["a-1-A-T", "b-2-C-G", "c-3-G-A", "d-4-T-C"] is four. They are
    told apart by the second element, which is a position in the tuple form.
    """
    if isinstance(v, (str, dict)):
        return True
    if isinstance(v, (tuple, list)) and len(v) == 4:
        return isinstance(v[1], int) or (
            isinstance(v[1], str) and v[1].isdigit())
    return False


class SpliceAI:
    """A loaded SpliceAI ensemble plus, optionally, SAI-10k-calc post-processing."""

    def __init__(self, reference, annotation=None, models=None, transcripts=None,
                 distance=500, mask=0, threads=None, cache_size=256,
                 thresholds=None, gff=None, genes=None, protein=True):
        """
        reference   : indexed FASTA.
        models      : directory holding spliceai1.h5 .. spliceai5.h5. Required.
        protein     : also predict the protein consequence of each event
                      (sequence, premature stop, NMD). Needs `transcripts`.
        annotation  : "grch37"/"grch38", a SpliceAI annotation table, or a
                      DataFrame. Required unless gff= is given: there is no
                      default build, because silently scoring against the wrong
                      one returns empty results rather than an error.
        distance    : window each side of the variant, default 500. Costs ~12%
                      more than 50 and sees ten times as much sequence.
        transcripts : genePred table; enables the SAI-10k-calc classification.
        gff         : a single GFF3/GTF used to derive *both* of the above. Use
                      this instead of annotation= and transcripts= so the two
                      cannot drift apart between releases.
        genes       : optional iterable of gene or transcript IDs to keep.
        """
        if gff is not None:
            annotation, transcripts = self._from_gff(gff, genes)
        if annotation is None:
            raise TypeError(
                "SpliceAI() requires annotation= ('grch37', 'grch38', a table "
                "path, or a DataFrame), or gff=. There is no default build: "
                "scoring GRCh38 data against a GRCh37 table returns empty "
                "results rather than failing.")
        if models is None:
            raise TypeError(
                "SpliceAI() requires models=<directory holding spliceai1..5.h5>. "
                "The weights are never discovered implicitly, so a run always "
                "records which ones it used.")
        if not Path(models).is_dir():
            raise FileNotFoundError("models directory not found: {}".format(models))
        self.distance = int(distance)
        self.mask = int(mask)
        self.annotator = Annotator(reference, annotation)
        self.ensemble = Ensemble(models, threads=threads)
        log.info("scoring window: -D %d (%d nt per pass)", self.distance,
                 10000 + 2 * self.distance + 1)
        self._cache = _RefCache(self.ensemble, limit=cache_size)
        from .sai10k.classify import Thresholds
        self.thresholds = Thresholds.make(thresholds)
        self.protein = protein
        self.exons = None
        if isinstance(transcripts, pd.DataFrame):
            self.exons = transcripts
        elif transcripts is not None:
            from .sai10k import build_exon_table, load_genepred
            self.exons = build_exon_table(load_genepred(transcripts))

    @staticmethod
    def _from_gff(gff, genes=None):
        """Parse one GFF3/GTF into (annotation DataFrame, exon-table DataFrame)."""
        from .sai10k.exons import build_exon_table
        from .transcripts import parse, to_genepred_table, to_spliceai_table

        recs = parse(gff)
        if genes:
            wanted = {str(g) for g in genes}
            recs = {t: r for t, r in recs.items()
                    if r["name"] in wanted or t in wanted or t.split(".")[0] in wanted}
        if not recs:
            raise ValueError("no transcripts parsed from {}".format(gff))
        ann = pd.DataFrame(to_spliceai_table(recs, name_field="transcript",
                                             name_prefix="RefSeqTx-"))
        gp = pd.DataFrame(to_genepred_table(recs))
        for c in ("txStart", "txEnd", "cdsStart", "cdsEnd", "exonCount"):
            gp[c] = gp[c].astype(int)
        return ann, build_exon_table(gp)


    # -- scoring ---------------------------------------------------------
    def score(self, variants, raw_scores=None, sai10k=None, include_unscored=False):
        """Score variants and return one row per (variant, transcript, allele).

        raw_scores : include the REF/ALT probabilities behind each delta score.
                     On by default - they cost nothing extra, since both passes
                     are computed anyway, and SAI-10k-calc's improved partial
                     deletion/retention calls need them.
        sai10k     : add the splicing-aberration classification. Defaults to
                     True when this object was built with `transcripts=`.
        """
        if sai10k is None:
            sai10k = self.exons is not None
        if sai10k and self.exons is None:
            raise ValueError("sai10k=True needs SpliceAI(transcripts=...)")
        if raw_scores is None:
            raw_scores = True

        if _is_single_variant(variants):
            variants = [variants]

        rows = []
        total = len(variants)
        every = max(1, total // 10)
        t0 = time.perf_counter()
        log.info("scoring %d variant%s", total, "" if total == 1 else "s")
        for i, raw in enumerate(variants, start=1):
            if total > 20 and i % every == 0:
                done = time.perf_counter() - t0
                log.info("  %d/%d  (%.1f variants/s, ~%.0fs left)", i, total,
                         i / done, done / i * (total - i))
            v = parse_variant(raw)
            scores = get_delta_scores(v, self.annotator, self.ensemble, self.distance,
                                      self.mask, ref_cache=self._cache, raw_scores=raw_scores)
            if not scores:
                log.debug("  %s-%s-%s: no transcript overlap or skipped",
                          v.chrom, v.pos, v.ref)
                if include_unscored:
                    rows.append({"#CHROM": v.chrom, "POS": v.pos, "REF": v.ref,
                                 "ALT": ",".join(v.alts), "ALLELE": None, "SYMBOL": None,
                                 "scored": False})
                continue
            for s in scores:
                rows.append(self._row(v, s, raw_scores))
        df = pd.DataFrame(rows)
        log.info("scored %d variants -> %d rows in %.1fs (%.2f variants/s)",
                 total, len(df), time.perf_counter() - t0,
                 total / max(time.perf_counter() - t0, 1e-9))
        if df.empty or not sai10k:
            return df
        return self.classify(df)

    def _row(self, v, score_string, raw_scores):
        parts = score_string.split("|")
        names = ["ALLELE", "SYMBOL"] + DELTA_COLS + (RAW_COLS if raw_scores else [])
        rec = {"#CHROM": v.chrom, "POS": v.pos, "ID": "{}-{}-{}-{}".format(
            v.chrom, v.pos, v.ref, parts[0]), "REF": v.ref, "ALT": parts[0],
            "QUAL": ".", "FILTER": ".", "scored": True}
        for k, val in zip(names, parts):
            rec[k] = val if k in ("ALLELE", "SYMBOL") else pd.to_numeric(val, errors="coerce")
        return rec

    def classify(self, scored):
        """Run SAI-10k-calc classification over an already-scored table."""
        from .sai10k import classify as _classify
        if self.exons is None:
            raise ValueError("classification needs SpliceAI(transcripts=...)")
        usable = scored[scored.get("scored", True).astype(bool)].copy()
        if usable.empty:
            return scored
        t = time.perf_counter()
        out = _classify(usable, self.exons, self.thresholds)
        called = int((out["Any_splicing_aberration"] == "YES").sum())
        log.info("classified %d rows, %d with a predicted aberration (%.1fs)",
                 len(out), called, time.perf_counter() - t)
        if self.protein:
            from .sai10k.protein import predict as _protein
            t = time.perf_counter()
            out = _protein(out, self.exons, self.annotator.ref_fasta)
            ptc = int(out["PTC_codon"].notna().sum())
            log.info("protein: %d predictions, %d with a premature stop (%.1fs)",
                     int(out["Protein_consequence"].notna().sum()), ptc,
                     time.perf_counter() - t)
        return out.reset_index(drop=True)


def score_variants(variants, reference, annotation=None, models=None, transcripts=None,
                   distance=500, mask=0, threads=None, gff=None,
                   genes=None, **kw):
    """One-shot convenience wrapper. Builds an engine, scores, throws it away.

    Loading the models takes about a second, so prefer reusing a `SpliceAI`
    instance when scoring more than one batch.
    """
    sai = SpliceAI(reference, annotation, models, transcripts, distance, mask,
                   threads, gff=gff, genes=genes)
    return sai.score(variants, **kw)
