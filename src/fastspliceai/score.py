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
# a position is worth reporting in the full profile if either class reaches this
MIN_SCORE = 0.01
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
    """Put y_alt back on the reference coordinate grid.

    Insertions collapse to their maximum, deletions and MNVs pad with zeros, so
    every alternate row lines up with the reference row it is subtracted from.
    """
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
    if ref_len > 1 and alt_len > 1:                       # MNV / delins
        return np.concatenate(
            [
                y_alt[: cov // 2],
                y_alt[cov // 2 : cov // 2 + alt_len].max(axis=0, keepdims=True),
                np.zeros((ref_len - 1, 3), y_alt.dtype),
                y_alt[cov // 2 + alt_len :],
            ]
        )
    return y_alt


# a rival within this fraction of the winner makes the predicted position
# ambiguous; sites nearer than RIVAL_EXCLUDE_BP are the same peak, not a rival
RIVAL_RATIO = 0.8
RIVAL_EXCLUDE_BP = 10


def rival_margins(y_ref, y_alt):
    """How clear-cut is each predicted position?

    A delta score is a maximum over the window, so it says nothing about what
    came second. Two cryptic donors at 0.44 and 0.42 give the same DS_DG as one
    at 0.44 with nothing behind it, but the predicted outcome is far less
    certain in the first case.

    Returns the runner-up as a fraction of the winner for each of the four
    events, excluding positions within RIVAL_EXCLUDE_BP of the winner - the
    model is smooth, so neighbours of a peak are the same peak.

    Two numpy passes over arrays we already hold; no measurable cost.
    """
    import numpy as np

    out = {}
    idx = np.arange(len(y_ref))
    for name, delta in (("AG", y_alt[:, 1] - y_ref[:, 1]),
                        ("AL", y_ref[:, 1] - y_alt[:, 1]),
                        ("DG", y_alt[:, 2] - y_ref[:, 2]),
                        ("DL", y_ref[:, 2] - y_alt[:, 2])):
        win = int(delta.argmax())
        best = float(delta[win])
        if best <= 0:
            out[name] = 0.0
            continue
        far = np.abs(idx - win) > RIVAL_EXCLUDE_BP
        runner = float(delta[far].max()) if far.any() else 0.0
        out[name] = round(max(runner, 0.0) / best, 3)
    return out


# a candidate outcome must gain at least this much, and be a separate peak
CANDIDATE_MIN_GAIN = 0.1
CANDIDATE_MAX = 6


def candidate_sites(y_ref, y_alt, cov, threshold=CANDIDATE_MIN_GAIN):
    """Every splice site the variant plausibly creates or destroys.

    The four delta scores report only the strongest of each kind. This returns
    each distinct peak above `threshold`, so a variant with three competing
    cryptic acceptors yields three candidates rather than one.

    Cheap: numpy over arrays already held, no per-position Python.
    """
    import numpy as np

    half = cov // 2
    out = []
    for kind, delta in (("acceptor_gain", y_alt[:, 1] - y_ref[:, 1]),
                        ("acceptor_loss", y_ref[:, 1] - y_alt[:, 1]),
                        ("donor_gain", y_alt[:, 2] - y_ref[:, 2]),
                        ("donor_loss", y_ref[:, 2] - y_alt[:, 2])):
        d = np.asarray(delta)
        peaks = np.flatnonzero(d >= threshold)
        if not len(peaks):
            continue
        # keep the local maximum of each run of adjacent positions
        for group in np.split(peaks, np.flatnonzero(np.diff(peaks) > RIVAL_EXCLUDE_BP) + 1):
            i = int(group[np.argmax(d[group])])
            out.append({"kind": kind, "offset": i - half, "score": round(float(d[i]), 3)})
    out.sort(key=lambda c: -c["score"])
    return out[:CANDIDATE_MAX]


def all_non_zero_scores(y_ref, y_alt, pos, cov, minus, threshold=MIN_SCORE):
    """Every position in the window where a splice signal is present.

    The four delta scores collapse the whole window to its maxima, so a second
    cryptic site is invisible. This keeps every position whose reference or
    alternate acceptor/donor probability reaches `threshold`, with genomic
    coordinates.

    Not free - a Python pass over up to 2*dist+1 positions per variant - so it
    is opt-in.
    """
    import numpy as np

    keep = np.nonzero(
        (y_ref[:, 1:].max(axis=1) >= threshold) | (y_alt[:, 1:].max(axis=1) >= threshold)
    )[0]
    half = cov // 2
    out = []
    for i in keep:
        offset = int(i) - half
        out.append({
            "pos": pos + (-offset if minus else offset),
            "offset": offset,
            "RA": round(float(y_ref[i, 1]), 3), "RD": round(float(y_ref[i, 2]), 3),
            "AA": round(float(y_alt[i, 1]), 3), "AD": round(float(y_alt[i, 2]), 3),
        })
    return out


def scores_for_inserted_bases(y_ref, y_alt_raw, pos, cov, ref, alt):
    """Per-base scores for the bases an insertion adds.

    The alignment step max-pools inserted bases into one position so the rows
    stay comparable, which discards the detail. This recovers it.
    """
    if not (len(ref) == 1 and len(alt) > 1):
        return []
    half = cov // 2
    out = []
    for k in range(len(alt)):
        row = y_alt_raw[half + k]
        out.append({
            "pos": pos + k, "base": alt[k],
            "RA": round(float(y_ref[min(half + k, len(y_ref) - 1), 1]), 3),
            "RD": round(float(y_ref[min(half + k, len(y_ref) - 1), 2]), 3),
            "AA": round(float(row[1]), 3), "AD": round(float(row[2]), 3),
        })
    return out


def get_delta_scores(record, ann, ensemble, dist_var=50, mask=0, ref_cache=None,
                     raw_scores=False, precision=2, all_scores=False,
                     extras=None):
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

            dist_ann = ann.dist_to_boundaries(i, pos)
            minus = ann.strands[i] == "-"
            x_ref = _pad_window(seq, wid, dist_ann)
            x_alt = x_ref[: wid // 2] + alt + x_ref[wid // 2 + len(ref) :]

            y_ref = ref_cache.get_ref(x_ref, minus)
            y_alt = _forward(ensemble, x_alt, minus)

            y_alt_raw = y_alt
            y_alt = _align_indel(y_alt, cov, len(ref), len(alt))
            out.append(_format(alt, gene, y_ref, y_alt, cov, dist_ann, mask,
                               raw_scores, precision))
            if extras is not None:
                extras.append({
                    "rival": rival_margins(y_ref, y_alt),
                    "candidates": candidate_sites(y_ref, y_alt, cov),
                    "ALL_NON_ZERO_SCORES": (
                        all_non_zero_scores(y_ref, y_alt, pos, cov, minus)
                        if all_scores else None),
                    "SCORES_FOR_INSERTED_BASES":
                        scores_for_inserted_bases(y_ref, y_alt_raw, pos, cov, ref, alt),
                })
    return out


def _format(alt, gene, y_ref, y_alt, cov, dist_ann, mask, raw_scores=False,
            precision=2):
    da, dd = y_alt[:, 1] - y_ref[:, 1], y_alt[:, 2] - y_ref[:, 2]
    i_pa, i_na = int(da.argmax()), int((-da).argmax())
    i_pd, i_nd = int(dd.argmax()), int((-dd).argmax())
    half = cov // 2
    bdry = dist_ann[2]
    m_pa = mask and (i_pa - half == bdry)
    m_na = mask and (i_na - half != bdry)
    m_pd = mask and (i_pd - half == bdry)
    m_nd = mask and (i_nd - half != bdry)
    fmt = "{{:.{}f}}".format(precision)
    line = "|".join(["{}", "{}"] + [fmt] * 4 + ["{}"] * 4).format(
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
            fmt.format(v)
            for v in (
                y_ref[i_pa, 1], y_alt[i_pa, 1],
                y_ref[i_na, 1], y_alt[i_na, 1],
                y_ref[i_pd, 2], y_alt[i_pd, 2],
                y_ref[i_nd, 2], y_alt[i_nd, 2],
            )
        )
    return line
