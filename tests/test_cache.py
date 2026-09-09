"""The cache must never serve a score computed under different inputs."""
from pathlib import Path

import pytest

from fastspliceai.cache import VariantCache, fingerprint

ROWS = [{"#CHROM": "chr9", "POS": 128504906, "REF": "T", "ALT": "G", "DS_DL": 0.99}]


@pytest.fixture
def db(tmp_path):
    return tmp_path / "scores.db"


def _fp(tmp_path, distance=500, mask=0, contents=b"a"):
    ann = tmp_path / "ann.txt"
    ann.write_bytes(contents)
    return fingerprint(annotation=ann, transcripts=None, models=None,
                       distance=distance, mask=mask), ann


def test_round_trip(db, tmp_path):
    fp, _ = _fp(tmp_path)
    c = VariantCache(db, fp)
    assert c.get("chr9-128504906-T-G") is None and c.misses == 1
    c.put("chr9-128504906-T-G", ROWS)
    got = c.get("chr9-128504906-T-G")
    assert got == ROWS and c.hits == 1
    c.close()


def test_survives_reopen(db, tmp_path):
    fp, _ = _fp(tmp_path)
    c = VariantCache(db, fp); c.put("v", ROWS); c.close()
    again = VariantCache(db, fp)
    assert again.get("v") == ROWS
    again.close()


def test_distance_change_invalidates(db, tmp_path):
    fp500, ann = _fp(tmp_path, distance=500)
    c = VariantCache(db, fp500); c.put("v", ROWS); c.close()

    fp50 = fingerprint(annotation=ann, transcripts=None, models=None,
                       distance=50, mask=0)
    assert fp50 != fp500
    other = VariantCache(db, fp50)
    assert other.get("v") is None, "a -D 50 query was served a -D 500 result"
    assert other.stale_rows() == 1
    other.close()


def test_annotation_change_invalidates(db, tmp_path):
    fp_a, ann = _fp(tmp_path, contents=b"first")
    c = VariantCache(db, fp_a); c.put("v", ROWS); c.close()

    ann.write_bytes(b"second and longer")          # size changes -> fingerprint changes
    fp_b = fingerprint(annotation=ann, transcripts=None, models=None,
                       distance=500, mask=0)
    assert fp_b != fp_a
    other = VariantCache(db, fp_b)
    assert other.get("v") is None, "served a score from a different transcript set"
    other.close()


def test_purge_stale_keeps_current(db, tmp_path):
    fp_a, ann = _fp(tmp_path, contents=b"first")
    c = VariantCache(db, fp_a); c.put("old", ROWS); c.close()
    ann.write_bytes(b"second")
    fp_b = fingerprint(annotation=ann, transcripts=None, models=None,
                       distance=500, mask=0)
    c = VariantCache(db, fp_b)
    c.put("new", ROWS)
    assert c.stale_rows() == 1
    assert c.purge_stale() == 1
    assert c.stale_rows() == 0
    assert c.get("new") == ROWS
    c.close()


def test_empty_result_is_cached(db, tmp_path):
    """A variant outside every transcript must not be rescored every time."""
    fp, _ = _fp(tmp_path)
    c = VariantCache(db, fp)
    c.put("chr1-1-A-T", [])
    assert c.get("chr1-1-A-T") == []      # a hit, not a miss
    assert c.hits == 1 and c.misses == 0
    c.close()


def test_disabled_cache_is_a_noop(tmp_path):
    c = VariantCache(None, "fp")
    c.put("v", ROWS)
    assert c.get("v") is None and len(c) == 0
    c.close()


def test_empty_cache_is_truthy(db, tmp_path):
    """__len__ makes an empty object falsy; `if cache:` must still work."""
    fp, _ = _fp(tmp_path)
    c = VariantCache(db, fp)
    assert len(c) == 0
    assert bool(c) is True, "an empty enabled cache must not be falsy"
    assert bool(VariantCache(None, fp)) is False
    c.close()
