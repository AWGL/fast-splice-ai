"""The SAI-10k-calc port must reproduce the R implementation's own example output."""
import re
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent

from fastspliceai.sai10k import build_exon_table, classify, load_genepred, parse_spliceai_vcf  # noqa: E402

GOLDEN_COLS = [
    "strand", "Used_RefSeq_Transcript", "Any_splicing_aberration", "bp_5prime", "bp_3prime",
    "Partial_intron_retention", "Partial_exon_deletion", "Pseudoexon_activation",
    "Pseudoexon_start", "Pseudoexon_end", "Pseudoexon_intron", "Pseudoexon_frameshift",
    "Increased_exon_inclusion", "Exon_with_increased_inclusion", "Exon_skipping",
    "Lost_exons", "Exon_skipping_frameshift", "Retained_intron_size", "Intron_retention",
    "Retained_intron", "Intron_retention_frameshift", "Gained_exon_size",
]


def _norm(s):
    s = s.astype(str).str.strip().replace({"nan": "", "None": "", "NA": "", "<NA>": ""})
    return s.map(lambda x: str(int(float(x))) if re.fullmatch(r"-?\d+(\.0+)?", x) else x)


@pytest.fixture(scope="module")
def compared():
    ex = build_exon_table(load_genepred(HERE / "data" / "refseq_brca_hg19.txt"))
    got = classify(parse_spliceai_vcf(HERE / "data" / "sai10k_example.vcf"), ex)
    gold = pd.read_csv(HERE / "data" / "sai10k_example.golden.tsv", sep="\t", dtype=str)
    got["POS"] = got["POS"].astype(str)
    return got.set_index(["POS", "ID"]), gold.set_index(["POS", "ID"])


def test_row_count(compared):
    got, gold = compared
    assert len(got) == len(gold) == 10


@pytest.mark.parametrize("col", GOLDEN_COLS)
def test_column_matches_r_implementation(col, compared):
    got, gold = compared
    a, b = _norm(got[col].reindex(gold.index)), _norm(gold[col])
    bad = gold.index[a != b]
    assert not len(bad), "{}: {} of {} rows differ, e.g. {} got={!r} gold={!r}".format(
        col, len(bad), len(gold), bad[0], a.loc[bad[0]], b.loc[bad[0]])


def test_exon_table_orientation():
    """eNum follows transcript order, so exon 1 is right-most on the minus strand."""
    ex = build_exon_table(load_genepred(HERE / "data" / "refseq_brca_hg19.txt"))
    b1 = ex[ex.name == "NM_007294.4"].sort_values("eNum")
    assert b1.iloc[0].strand == -1
    assert b1.iloc[0].eStart > b1.iloc[-1].eStart
    b2 = ex[ex.name == "NM_000059.4"].sort_values("eNum")
    assert b2.iloc[0].strand == 1
    assert b2.iloc[0].eStart < b2.iloc[-1].eStart


def _one_transcript_exons():
    """A + strand transcript with exons 1001-1100, 2001-2100 and 3001-3100 (genePred: 0-based starts)."""
    tx = pd.DataFrame([{
        "name": "NM_TEST.1", "chrom": "chr1", "strand": "+", "txStart": 1000, "txEnd": 3100,
        "cdsStart": 1000, "cdsEnd": 3100, "exonCount": 3, "exonStarts": "1000,2000,3000,",
        "exonEnds": "1100,2100,3100,", "name2": "TEST", "exonFrames": "0,1,2,"}])
    return build_exon_table(tx)


@pytest.mark.parametrize("pos, ref, d_from_exon, exon", [
    (1994, "A", 7, 2),                 # SNV before exon 2 (2001 - 1994): unchanged
    (1200, "A" * 794, 8, 2),           # deletion of 1201-1993 in intron 1, next to exon 2 (POS alone: 100 bp past exon 1)
    (1990, "A" * 20, 0, 2),            # deletion reaching into exon 2
    (2150, "A" * 10, 50, 2),           # deletion after exon 2 measured from its start, as before
])
def test_deletion_distance_from_its_nearest_base(pos, ref, d_from_exon, exon):
    from fastspliceai.sai10k.classify import _closest_exon

    variants = pd.DataFrame([{"#CHROM": "chr1", "POS": pos, "REF": ref, "ALT": ref[0], "SYMBOL": "RefSeqTx-NM_TEST.1",
                              "ID": "v1"}])
    got = _closest_exon(variants, _one_transcript_exons()).iloc[0]

    assert (int(got["d_from_exon"]), int(got["eNum"])) == (d_from_exon, exon)


def test_partial_shift_outcomes_follow_their_own_site():
    # a 300 bp acceptor shift out of the exon (too far for a partial intron
    # retention) and a 10 bp donor shift into it (the partial exon deletion)
    from fastspliceai.sai10k.classify import _outcomes
    row = {"Pseudoexon_activation": "NO", "Exon_skipping": "NO", "Intron_retention": "NO",
           "Partial_exon_deletion": "YES", "Partial_intron_retention": "NO",
           "bp_5prime": -300, "bp_3prime": -10, "DS_AG": 0.2, "DS_DG": 0.7}
    outs = _outcomes(row)
    assert [(o["event"], o["bp"], o["frameshift"]) for o in outs] == [("donor shift", 10, True)]

    # both sites shifting into the exon: two separate outcomes, each its own size
    row.update({"bp_5prime": 6, "bp_3prime": -6})
    outs = _outcomes(row)
    assert sorted((o["event"], o["bp"]) for o in outs) == [("acceptor shift", 6), ("donor shift", 6)]
