"""Protein consequence prediction, checked against the R implementation.

The R tool's example output is hg19 BRCA1/BRCA2, so these need an hg19 FASTA
covering those two loci. It is 75 MB once N-padded to real coordinates, so it is
not committed; build it with:

    python tests/make_hg19_fixture.py
"""
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "data" / "hg19_brca.fa"

from fastspliceai.sai10k import build_exon_table, classify, load_genepred, parse_spliceai_vcf  # noqa: E402
from fastspliceai.sai10k.protein import exons_for_outcome, predict, translate  # noqa: E402


def test_translate_stops_at_first_stop():
    assert translate("ATGAAATAG") == ("MK*", True)
    assert translate("ATGAAAAAA") == ("MKK", False)
    assert translate("ATG") == ("M", False)


def test_translate_handles_ambiguous_bases():
    aa, stopped = translate("ATGNNNAAA")
    assert aa == "M" and not stopped


@pytest.mark.skipif(not FIXTURE.exists(),
                    reason="hg19 fixture absent; build with tests/make_hg19_fixture.py")
def test_matches_r_implementation():
    ex = build_exon_table(load_genepred(HERE / "data" / "refseq_brca_hg19.txt"))
    d = predict(classify(parse_spliceai_vcf(HERE / "data" / "sai10k_example.vcf"), ex),
                ex, str(FIXTURE))
    gold = pd.read_csv(HERE / "data" / "sai10k_example.golden.tsv", sep="\t",
                       dtype=str).set_index("ID")
    aa_cols = [c for c in gold.columns if c.endswith("_aaseq")]
    checked = 0
    for _, r in d.iterrows():
        theirs = {gold.loc[r["ID"], c] for c in aa_cols
                  if gold.loc[r["ID"], c] not in ("", "-", "NA")}
        if not theirs:
            continue
        ours = r["Protein_consequence"] or r["Protein_note"]
        assert ours in theirs, "{}: ours={!r} R={}".format(r["ID"], ours, sorted(theirs))
        checked += 1
    assert checked >= 4


@pytest.mark.skipif(not FIXTURE.exists(), reason="hg19 fixture absent")
def test_reports_premature_stop_and_nmd():
    ex = build_exon_table(load_genepred(HERE / "data" / "refseq_brca_hg19.txt"))
    d = predict(classify(parse_spliceai_vcf(HERE / "data" / "sai10k_example.vcf"), ex),
                ex, str(FIXTURE))
    skip = d[d["ID"] == "BRCA1_c.5467+5G>C"].iloc[0]
    assert skip["Protein_consequence"] == "LGT[QLGRCVRHLW*]"
    assert skip["PTC_codon"] == 1813
    # the stop lands in the final exon, so it should escape nonsense-mediated decay
    assert skip["NMD_predicted"] == "NO"
    assert "premature stop at codon 1813" in skip["Consequence"]


class _FakeTx:
    """Three 30bp exons on the plus strand, at 101-130, 201-230, 301-330."""
    name, chrom, strand = "tx", "chr1", "+"
    cdsStart, cdsEnd = 100, 330
    exons = [(101, 130), (201, 230), (301, 330)]


def _row(**kw):
    base = {"Lost_exons": None, "Lost_exons_combined_size": float("nan"),
            "POS": 205, "Exon_skipping": "NO", "Pseudoexon_activation": "NO",
            "Intron_retention": "NO", "bp_5prime": 0, "bp_3prime": 0}
    base.update(kw)
    return pd.Series(base)


def test_exons_for_outcome_rebuilds_each_alternative():
    """Every enumerated outcome must be reconstructible, not just the called one.

    Judging alternatives by bp %% 3 alone misses an in-frame outcome that
    carries a stop codon, so each one has to yield a real exon list.
    """
    tx, ex = _FakeTx(), list(_FakeTx.exons)
    row = _row()

    skip = exons_for_outcome({"event": "exon skipping", "bp": 30, "exon": 2}, row, tx, ex)
    assert skip == [(101, 130), (301, 330)]

    # an acceptor opens the exon on the plus strand: exon 2 starts later
    shift = exons_for_outcome({"event": "acceptor shift", "bp": 6, "exon": 2,
                               "site": "acceptor", "new_pos": 207}, row, tx, ex)
    assert shift == [(101, 130), (207, 230), (301, 330)]

    # a donor closes it
    shift = exons_for_outcome({"event": "donor shift", "bp": 6, "exon": 2,
                               "site": "donor", "new_pos": 224}, row, tx, ex)
    assert shift == [(101, 130), (201, 224), (301, 330)]

    pe = exons_for_outcome({"event": "pseudoexon", "bp": 21, "exon": 2,
                            "start": 250, "end": 270}, row, tx, ex)
    assert pe == [(101, 130), (201, 230), (250, 270), (301, 330)]

    # no geometry -> fall back to the row's own called event
    assert exons_for_outcome({"event": "exon skipping", "bp": 30}, row, tx, ex) is None


def test_exons_for_outcome_is_strand_aware():
    class Minus(_FakeTx):
        strand = "-"
        exons = [(301, 330), (201, 230), (101, 130)]     # transcript order

    tx, ex = Minus(), list(Minus.exons)
    row = _row()
    # on the minus strand the acceptor is the higher coordinate
    shift = exons_for_outcome({"event": "acceptor shift", "bp": 6, "exon": 2,
                               "site": "acceptor", "new_pos": 224}, row, tx, ex)
    assert shift == [(301, 330), (201, 224), (101, 130)]
    shift = exons_for_outcome({"event": "donor shift", "bp": 6, "exon": 2,
                               "site": "donor", "new_pos": 207}, row, tx, ex)
    assert shift == [(301, 330), (207, 230), (101, 130)]


def test_impossible_shift_is_rejected():
    tx, ex = _FakeTx(), list(_FakeTx.exons)
    # boundary moved past the far end of the exon
    assert exons_for_outcome({"event": "acceptor shift", "bp": 99, "exon": 2,
                              "site": "acceptor", "new_pos": 999}, _row(), tx, ex) is None
    assert exons_for_outcome({"event": "acceptor shift", "bp": 4, "exon": 9,
                              "site": "acceptor", "new_pos": 207}, _row(), tx, ex) is None


def _synthetic_gene(tmp_path):
    """A 3-exon plus-strand gene whose codons line up with the exon boundaries.

    exon1 101-130  ATG + 9x GCT          M A A A A A A A A A
    exon2 201-230  10x AAA               K K K K K K K K K K
    exon3 301-360  19x CCT + TAA         P x19 *

    Intron 1 also holds a 30bp in-frame pseudoexon at 151-180 carrying a stop.

    Exon 2 is 30bp, so skipping it is in frame and removes ten lysines without
    touching the stop codon: the protein is shorter but perfectly good.
    """
    seq = ["N"] * 400
    def put(start, s):
        for k, c in enumerate(s):
            seq[start - 1 + k] = c
    put(101, "ATG" + "GCT" * 9)
    put(201, "AAA" * 10)
    put(301, "CCT" * 19 + "TAA")
    # a 30bp in-frame pseudoexon sitting in intron 1, carrying a stop codon
    put(151, "GGGGGGTAA" + "GGG" * 7)
    fa = tmp_path / "synth.fa"
    fa.write_text(">chr1\n" + "".join(seq) + "\n")
    import pysam
    pysam.faidx(str(fa))
    return fa


def test_in_frame_outcome_keeping_the_native_stop_is_not_disruptive(tmp_path):
    """The negative control: in-frame *and* no new stop must come back clean.

    An in-frame deletion shortens the peptide while still terminating on the
    annotated stop codon. Judging that by "the mutant peptide is shorter than
    wild type" would call every in-frame deletion a premature stop, so the
    check is where the stop codon sits in the genome, not how far along it is.
    """
    from pyfaidx import Fasta
    from fastspliceai.sai10k.protein import _consequence, _spliced, _cds_offset, _genomic_at

    fa = Fasta(str(_synthetic_gene(tmp_path)), rebuild=False)
    tx, ex = _FakeTx(), [(101, 130), (201, 230), (301, 360)]
    tx.exons = ex
    tx.cdsStart, tx.cdsEnd = 100, 360
    wt_seq, _ = _spliced(fa, "chr1", ex, "+")
    off = _cds_offset(tx, ex, "+")
    wt_aa, stopped = translate(wt_seq[off:])
    assert wt_aa == "M" + "A" * 9 + "K" * 10 + "P" * 19 + "*" and stopped
    wt = (wt_aa, _genomic_at(ex, "+", off + (len(wt_aa) - 1) * 3))

    skipped = [(101, 130), (301, 360)]
    res = _consequence(fa, tx, ex, skipped, (205, "A", "A"), wt)
    assert res["disruptive"] is False, res
    assert "ptc" not in res, "native stop reported as premature: {}".format(res)

    # a smaller in-frame deletion is clean for the same reason
    res2 = _consequence(fa, tx, ex, [(101, 130), (201, 230), (304, 360)],
                        (205, "A", "A"), wt)
    assert res2["disruptive"] is False and "ptc" not in res2

    # positive control on the same gene: a 30bp pseudoexon is in frame too, but
    # it carries a stop codon, so it must come back disruptive
    with_pe = [(101, 130), (151, 180), (201, 230), (301, 360)]
    res3 = _consequence(fa, tx, ex, with_pe, (205, "A", "A"), wt)
    assert res3["disruptive"] is True, res3
    assert res3["ptc"] == 13, res3            # M A*9 G G * -> stop at codon 13
    assert res3["nmd"] == "YES"


def test_unreconstructable_outcome_is_unknown_not_clean():
    """An outcome we cannot rebuild must not be counted as harmless."""
    from fastspliceai.sai10k.protein import _consequence
    res = _consequence(None, _FakeTx(), [], None, None, ("M*", 1))
    assert res["disruptive"] is None


def test_lof_note_ties_truncating_to_the_counted_outcomes():
    """"in-frame but truncating" must clearly describe a *counted* outcome.

    Written as two bare parentheticals it reads as a coordinate fact and looks
    like it might be describing the outcome that was ignored.
    """
    import re
    src = (Path(__file__).resolve().parents[1] / "src" / "fastspliceai" /
           "sai10k" / "protein.py").read_text()
    note = re.search(r'detail = "\{\}/\{\} plausible outcomes disruptive".*?'
                     r'lof_note\.append\(detail\)', src, re.S).group(0)
    assert "of them in-frame but truncating" in note
    # the ignored count comes last and is separated from the counted set
    assert note.index("of them in-frame") < note.index("not counted")
    assert '"; {} weaker outcome' in note
