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
from fastspliceai.sai10k.protein import predict, translate  # noqa: E402


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
