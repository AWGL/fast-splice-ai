"""Delta-score computation.

Semantics follow SpliceAI v1.3.1 exactly (verified against its published test
vectors). The speed comes from how the work is scheduled, not from what is
computed:

* the reference pass is computed once per distinct (window, strand) and reused
  across every ALT allele and every transcript that shares it.

Sequences are deliberately submitted one at a time. ONNX Runtime's CPU
convolution parallelises over the batch dimension, so batching two 10kb
sequences halves the spatial parallelism without gaining anything: measured
35 ms/pass at batch 1 versus 169 ms/pass at batch 2 on a 10-core CPU.
"""
from __future__ import annotations

import logging

import numpy as np

from .engine import CONTEXT, one_hot

logger = logging.getLogger(__name__)
SKIP_ALT_CHARS = (".", "-", "*", "<", ">")
EMPTY = "{}|{}|.|.|.|.|.|.|.|."


class _RefCache(dict):
    """y_ref keyed on the exact reference window and strand."""

    def __init__(self, ensemble, limit=256):
        super().__init__()
        self.ensemble = ensemble
        self.limit = limit
        self.hits = self.misses = 0

    def get_ref(self, x_ref, minus):
        key = (x_ref, minus)
        hit = self.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        y = _forward(self.ensemble, x_ref, minus)
        if len(self) >= self.limit:
            self.pop(next(iter(self)))
        self[key] = y
        return y


def _forward(ensemble, seq, minus):
    """Score one sequence; returns (L, 3)."""
    x = one_hot(seq)[None]
    if minus:
        x = x[:, ::-1, ::-1]
    y = ensemble.predict(np.ascontiguousarray(x))
    if minus:
        y = y[:, :, ::-1]
    return np.ascontiguousarray(y[0].T)


def _pad_window(seq, wid, dist_ann):
    """Mask the window down to the transcript, as upstream does."""
    lo = max(wid // 2 + dist_ann[0], 0)
    hi = max(wid // 2 - dist_ann[1], 0)
    return "N" * lo + seq[lo : wid - hi] + "N" * hi


def _align_indel(y_alt, cov, ref_len, alt_len):
    """Put y_alt back on the reference coordinate grid."""
    if ref_len > 1 and alt_len == 1:                      # deletion
        del_len = ref_len - alt_len
        return np.concatenate(
            [y_alt[: cov // 2 + alt_len], np.zeros((del_len, 3), y_alt.dtype), y_alt[cov // 2 + alt_len :]]
        )
    if ref_len == 1 and alt_len > 1:                      # insertion
        return np.concatenate(
            [
                y_alt[: cov // 2],
                y_alt[cov // 2 : cov // 2 + alt_len].max(axis=0, keepdims=True),
                y_alt[cov // 2 + alt_len :],
            ]
        )
    return y_alt


def get_delta_scores(record, ann, ensemble, dist_var=50, mask=0, ref_cache=None,
                     raw_scores=False):
    """Return the list of SpliceAI INFO strings for one VCF record.

    With raw_scores=True, eight extra fields are appended in the layout
    SAI-10k-calc's REF/ALT mode expects:
    DS_AG_REF|DS_AG_ALT|DS_AL_REF|DS_AL_ALT|DS_DG_REF|DS_DG_ALT|DS_DL_REF|DS_DL_ALT
    Each delta score is the difference of its own ALT and REF pair.
    """
    out = []
    try:
        chrom, pos, ref, alts = record.chrom, record.pos, record.ref, list(record.alts)
    except (TypeError, AttributeError):
        logger.warning("Skipping record (bad input): %s", record)
        return out

    cov = 2 * dist_var + 1
    wid = CONTEXT + cov

    idxs = ann.transcripts(chrom, pos)
    if len(idxs) == 0:
        return out

    seq = ann.sequence(chrom, pos - wid // 2 - 1, pos + wid // 2)
    if len(seq) != wid:
        logger.warning("Skipping record (near chromosome end): %s", record)
        return out
    if seq[wid // 2 : wid // 2 + len(ref)].upper() != ref.upper():
        logger.warning("Skipping record (ref does not match fasta): %s", record)
        return out
    if len(ref) > 2 * dist_var:
        logger.warning("Skipping record (ref too long): %s", record)
        return out

    if ref_cache is None:
        ref_cache = _RefCache(ensemble)

    for alt in alts:
        alt = str(alt)
        for i in idxs:
            gene = ann.genes[i]
            if any(c in alt for c in SKIP_ALT_CHARS):
                continue
            if len(ref) > 1 and len(alt) > 1:            # MNP / complex: not scored
                blank = EMPTY + ("|." * 8 if raw_scores else "")
                out.append(blank.format(alt, gene))
                continue

            dist_ann = ann.dist_to_boundaries(i, pos)
            minus = ann.strands[i] == "-"
            x_ref = _pad_window(seq, wid, dist_ann)
            x_alt = x_ref[: wid // 2] + alt + x_ref[wid // 2 + len(ref) :]

            y_ref = ref_cache.get_ref(x_ref, minus)
            y_alt = _forward(ensemble, x_alt, minus)

            y_alt = _align_indel(y_alt, cov, len(ref), len(alt))
            out.append(_format(alt, gene, y_ref, y_alt, cov, dist_ann, mask, raw_scores))
    return out


def _format(alt, gene, y_ref, y_alt, cov, dist_ann, mask, raw_scores=False):
    da, dd = y_alt[:, 1] - y_ref[:, 1], y_alt[:, 2] - y_ref[:, 2]
    i_pa, i_na = int(da.argmax()), int((-da).argmax())
    i_pd, i_nd = int(dd.argmax()), int((-dd).argmax())
    half = cov // 2
    bdry = dist_ann[2]
    m_pa = mask and (i_pa - half == bdry)
    m_na = mask and (i_na - half != bdry)
    m_pd = mask and (i_pd - half == bdry)
    m_nd = mask and (i_nd - half != bdry)
    line = "{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}".format(
        alt,
        gene,
        da[i_pa] * (not m_pa),
        -da[i_na] * (not m_na),
        dd[i_pd] * (not m_pd),
        -dd[i_nd] * (not m_nd),
        i_pa - half,
        i_na - half,
        i_pd - half,
        i_nd - half,
    )
    if raw_scores:
        # raw probabilities at each argmax; reported unmasked by definition
        line += "|" + "|".join(
            "{:.2f}".format(v)
            for v in (
                y_ref[i_pa, 1], y_alt[i_pa, 1],
                y_ref[i_na, 1], y_alt[i_na, 1],
                y_ref[i_pd, 2], y_alt[i_pd, 2],
                y_ref[i_nd, 2], y_alt[i_nd, 2],
            )
        )
    return line
