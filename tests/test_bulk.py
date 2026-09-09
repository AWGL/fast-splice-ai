"""Bulk regression: score a VCF and compare against a golden file.

The golden file was produced by *stock* SpliceAI v1.3.1 (TensorFlow), so this
suite asserts that fast-splice-ai's fp32 path is byte-identical to upstream over
a large, varied variant set - SNVs, insertions and deletions, on a minus-strand
transcript, including positions where the delta score is zero.

Scale it up with:
    FASTSPLICEAI_BULK_N=10000 pytest tests/test_bulk.py
(generate a bigger VCF + golden with benchmarks/make_vcf.py and
 benchmarks/compare.py first).
"""
import os
from collections import namedtuple
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from fastspliceai import Annotator, Ensemble, get_delta_scores  # noqa: E402
from fastspliceai.score import _RefCache                        # noqa: E402

Variant = namedtuple("Variant", "chrom pos ref alts")
VCF = HERE / "data" / "bulk.vcf"
GOLDEN = HERE / "data" / "bulk.golden.vcf"
LIMIT = int(os.environ.get("FASTSPLICEAI_BULK_N", "40"))


def _read(path):
    """[(chrom, pos, ref, alt, spliceai_string_or_None)] in file order."""
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
            rows.append((f[0], int(f[1]), f[3], f[4], tag))
    return rows


pytestmark = pytest.mark.skipif(
    not (VCF.exists() and GOLDEN.exists()),
    reason="bulk fixtures absent; build them with benchmarks/make_vcf.py",
)


@pytest.fixture(scope="module")
def scored():
    ens = Ensemble(ROOT / "spliceai" / "models")
    ann = Annotator(ROOT / "tests" / "data" / "test.fa", "grch37")
    cache = _RefCache(ens)
    out = []
    for chrom, pos, ref, alt, expected in _read(GOLDEN)[:LIMIT]:
        got = get_delta_scores(
            Variant(chrom, pos, ref, alt.split(",")), ann, ens, dist_var=50, mask=0,
            ref_cache=cache,
        )
        out.append((chrom, pos, ref, alt, ",".join(got) if got else None, expected))
    return out


def test_bulk_matches_stock_spliceai(scored):
    bad = [r for r in scored if r[4] != r[5]]
    detail = "\n".join(
        "{}:{} {}>{}\n  stock {}\n  fast  {}".format(c, p, r, a, exp, got)
        for c, p, r, a, got, exp in bad[:10]
    )
    assert not bad, "{}/{} records differ from stock SpliceAI\n{}".format(
        len(bad), len(scored), detail
    )


def test_bulk_covers_indels(scored):
    kinds = {"snv": 0, "ins": 0, "del": 0}
    for _, _, ref, alt, _, _ in scored:
        first = alt.split(",")[0]
        kinds["snv" if len(ref) == len(first) == 1 else "ins" if len(first) > len(ref) else "del"] += 1
    assert kinds["snv"] and (kinds["ins"] or kinds["del"]), kinds
