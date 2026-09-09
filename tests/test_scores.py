"""fast-splice-ai must reproduce SpliceAI v1.3.1 output exactly."""
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

from fastspliceai import Annotator, Ensemble, get_delta_scores  # noqa: E402

Record = namedtuple("Record", "chrom pos ref alts")

# taken verbatim from the upstream test-suite (tests/test_delta_score.py)
CASES = [
    (Record("10", 94077, "A", ["C"]), "C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"),
    (Record("chr10", 94077, "A", ["C"]), "C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193"),
    (Record("10", 94555, "C", ["T"]), "T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"),
    (Record("chr10", 94555, "C", ["T"]), "T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0"),
]


@pytest.fixture(scope="module")
def engine():
    return Ensemble(ROOT / "spliceai" / "models")


@pytest.fixture(scope="module", params=["test.fa", "test_without_prefix.fa"])
def ann(request):
    return Annotator(ROOT / "tests" / "data" / request.param, "grch37")


@pytest.mark.parametrize("record,expected", CASES)
def test_matches_upstream(record, expected, ann, engine):
    assert get_delta_scores(record, ann, engine, dist_var=500, mask=0) == [expected]


def test_ref_cache_reused_across_alleles(ann, engine):
    from fastspliceai.score import _RefCache

    cache = _RefCache(engine)
    rec = Record("10", 94077, "A", ["C", "G", "T"])
    scores = get_delta_scores(rec, ann, engine, dist_var=500, mask=0, ref_cache=cache)
    assert len(scores) == 3
    assert cache.misses == 1 and cache.hits == 2, (cache.misses, cache.hits)


def test_valid_rewrite_is_length_agnostic(engine):
    import numpy as np

    from fastspliceai.engine import CONTEXT

    rng = np.random.default_rng(0)
    L = CONTEXT + 3000
    x = np.zeros((1, 4, L), np.float32)
    x[0, rng.integers(0, 4, L), np.arange(L)] = 1
    whole = engine.predict(x)[0]
    # a 500-position slice must score identically when given its own context
    sub = engine.predict(x[:, :, 1000 : 1000 + CONTEXT + 500])[0]
    assert np.abs(whole[:, 1000:1500] - sub).max() < 1e-5
