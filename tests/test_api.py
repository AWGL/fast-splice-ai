"""End-to-end: GFF -> annotation tables -> scores -> SAI-10k classification."""
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from fastspliceai import SpliceAI, parse_variant  # noqa: E402
from fastspliceai.transcripts import (  # noqa: E402
    parse, to_genepred_table, to_spliceai_table, write_genepred, write_table,
)

# TUBB8, the only transcript in the bundled test FASTA with full 10 kb context
TUBB8 = [(92893, 94054), (94555, 94665), (94744, 94852), (95122, 95529)]


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    d = tmp_path_factory.mktemp("tx")
    gff = d / "tubb8.gff3"
    with open(gff, "w") as f:
        f.write("##gff-version 3\n")
        f.write("chr10\t.\tmRNA\t92893\t95529\t.\t-\t.\t"
                "ID=rna-NM_TEST.1;gene=TUBB8;transcript_id=NM_TEST.1\n")
        for s, e in TUBB8:
            f.write("chr10\t.\texon\t{}\t{}\t.\t-\t.\tParent=rna-NM_TEST.1;"
                    "transcript_id=NM_TEST.1\n".format(s, e))
            f.write("chr10\t.\tCDS\t{}\t{}\t.\t-\t0\tParent=rna-NM_TEST.1;"
                    "transcript_id=NM_TEST.1\n".format(s, e))
    recs = parse(gff)
    ann, gp = d / "ann.txt", d / "genepred.txt"
    write_table(to_spliceai_table(recs, name_field="transcript", name_prefix="RefSeqTx-"), ann)
    write_genepred(to_genepred_table(recs), gp)
    return ann, gp


@pytest.fixture(scope="module")
def sai(tables):
    ann, gp = tables
    return SpliceAI(reference=ROOT / "tests" / "data" / "test.fa", annotation=ann,
                    models=ROOT / "spliceai" / "models", transcripts=gp, distance=500)


def test_gff_round_trip_exon_count(tables):
    ann, gp = tables
    assert sum(1 for _ in open(ann)) == 2                       # header + one transcript
    row = open(gp).read().strip().split("\n")[1].split("\t")
    assert row[0] == "NM_TEST.1" and int(row[7]) == len(TUBB8)


def test_score_returns_rows_for_each_allele(sai):
    df = sai.score(["chr10-94077-A-C", ("chr10", 94555, "C", "T")], sai10k=False)
    assert len(df) == 2
    assert set(df["ALT"]) == {"C", "T"}
    assert df["DS_AG"].notna().all() and df["DS_AG_ALT"].notna().all()
    # raw REF/ALT scores must reconstruct the delta scores
    for _, r in df.iterrows():
        assert abs((r["DS_AG_ALT"] - r["DS_AG_REF"]) - r["DS_AG"]) < 0.011


def test_multi_allelic_expands(sai):
    df = sai.score({"chrom": "chr10", "pos": 94077, "ref": "A", "alt": ["C", "G", "T"]},
                   sai10k=False)
    assert sorted(df["ALT"]) == ["C", "G", "T"]


def test_sai10k_columns_present(sai):
    df = sai.score(["chr10-94555-C-T"])
    for col in ("Any_splicing_aberration", "Exon_skipping", "Intron_retention",
                "Pseudoexon_activation", "Used_RefSeq_Transcript"):
        assert col in df.columns
    assert df["Any_splicing_aberration"].isin(["YES", "NO"]).all()


def test_unscored_variants(sai):
    df = sai.score(["chr10-70000-A-C"], sai10k=False)          # outside any transcript
    assert df.empty
    df = sai.score(["chr10-70000-A-C"], sai10k=False, include_unscored=True)
    assert len(df) == 1 and not df.iloc[0]["scored"]


def test_single_variant_shorthand(sai):
    assert len(sai.score("chr10-94077-A-C", sai10k=False)) == 1
    assert len(sai.score(("chr10", 94077, "A", "C"), sai10k=False)) == 1


@pytest.fixture(scope="module")
def sai_from_gff(tmp_path_factory):
    """The same transcript set, but derived from one GFF instead of two tables."""
    d = tmp_path_factory.mktemp("gff")
    gff = d / "tubb8.gff3"
    with open(gff, "w") as f:
        f.write("##gff-version 3\n")
        f.write("chr10\t.\tmRNA\t92893\t95529\t.\t-\t.\t"
                "ID=rna-NM_TEST.1;gene=TUBB8;transcript_id=NM_TEST.1\n")
        for s_, e_ in TUBB8:
            f.write("chr10\t.\texon\t{}\t{}\t.\t-\t.\tParent=rna-NM_TEST.1;"
                    "transcript_id=NM_TEST.1\n".format(s_, e_))
            f.write("chr10\t.\tCDS\t{}\t{}\t.\t-\t0\tParent=rna-NM_TEST.1;"
                    "transcript_id=NM_TEST.1\n".format(s_, e_))
    return SpliceAI(reference=ROOT / "tests" / "data" / "test.fa", gff=gff,
                    models=ROOT / "spliceai" / "models", distance=500)


def test_gff_constructor_enables_sai10k(sai_from_gff):
    df = sai_from_gff.score(["chr10-94555-C-T"])
    assert "Any_splicing_aberration" in df.columns
    assert sai_from_gff.exons is not None and len(sai_from_gff.exons) == len(TUBB8)


def test_gff_constructor_matches_two_table_path(sai, sai_from_gff):
    """One GFF must give exactly what the separate annotation + genePred pair gives."""
    variants = ["chr10-94555-C-T", "chr10-94077-A-C"]
    a = sai.score(variants).reset_index(drop=True)
    b = sai_from_gff.score(variants).reset_index(drop=True)
    cols = [c for c in ("DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG",
                        "DP_DL", "Any_splicing_aberration", "Exon_skipping",
                        "Intron_retention", "Lost_exons") if c in a.columns and c in b.columns]
    assert cols
    for c in cols:
        assert a[c].astype(str).tolist() == b[c].astype(str).tolist(), c


def test_gff_gene_filter(tmp_path):
    gff = tmp_path / "two.gff3"
    with open(gff, "w") as f:
        f.write("##gff-version 3\n")
        for tx, gene in (("NM_A.1", "GENE_A"), ("NM_B.1", "GENE_B")):
            f.write("chr10\t.\tmRNA\t92893\t95529\t.\t-\t.\t"
                    "ID=rna-{0};gene={1};transcript_id={0}\n".format(tx, gene))
            f.write("chr10\t.\texon\t92893\t94054\t.\t-\t.\tParent=rna-{0};"
                    "transcript_id={0}\n".format(tx))
    ann, ex = SpliceAI._from_gff(gff, genes=["GENE_A"])
    assert len(ann) == 1 and ann.iloc[0]["#NAME"] == "RefSeqTx-NM_A.1"
    assert set(ex["name"]) == {"NM_A.1"}


def test_bad_variant_rejected():
    with pytest.raises(ValueError):
        parse_variant("not-a-variant-at-all!")


def test_second_ensemble_with_different_threads_does_not_crash():
    """ORT's thread pool is process-wide; a second Ensemble must warn, not die."""
    import warnings

    from fastspliceai import Ensemble

    a = Ensemble(ROOT / "spliceai" / "models", threads=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = Ensemble(ROOT / "spliceai" / "models", threads=7)
    assert b.sessions and len(b.sessions) == len(a.sessions)
    # either the pool was shared (and warned) or per-session threads were used
    assert not caught or issubclass(caught[-1].category, RuntimeWarning)


def test_four_variants_are_not_mistaken_for_one(sai):
    """A list of exactly four variants must not be read as one (chrom,pos,ref,alt)."""
    from fastspliceai.api import _is_single_variant

    assert _is_single_variant(("chr10", 94077, "A", "C"))
    assert _is_single_variant(["chr10", "94077", "A", "C"])
    assert not _is_single_variant(["chr10-94077-A-C"] * 4)
    df = sai.score(["chr10-94077-A-C", "chr10-94555-C-T",
                    "chr10-94077-A-G", "chr10-94077-A-T"], sai10k=False)
    assert len(df) == 4


def test_tile_matches_windowed_scoring(sai):
    """Ensemble.tile() is public and documented; assert it agrees with predict()."""
    import numpy as np

    from fastspliceai.engine import CONTEXT

    rng = np.random.default_rng(0)
    L = CONTEXT + 4000
    x = np.zeros((4, L), np.float32)
    x[rng.integers(0, 4, L), np.arange(L)] = 1
    whole = sai.ensemble.predict(x[None])[0]
    tiled = sai.ensemble.tile(x, tile=1500)             # forces several tiles
    assert tiled.shape == whole.shape
    assert np.abs(tiled - whole).max() < 1e-5


def test_score_variants_one_shot(tables):
    """The convenience wrapper is exported and documented; make sure it runs."""
    from fastspliceai import score_variants

    ann, gp = tables
    df = score_variants(["chr10-94077-A-C"], reference=ROOT / "tests" / "data" / "test.fa",
                        annotation=ann, transcripts=gp,
                        models=ROOT / "spliceai" / "models", distance=500)
    assert len(df) == 1 and df.iloc[0]["ALT"] == "C"
