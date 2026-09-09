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
