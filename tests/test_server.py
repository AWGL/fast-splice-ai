"""HTTP layer and cache integration, with a stub engine.

Deliberately no real inference: this is testing request handling, cache
behaviour and error paths, which are the parts that break silently.
"""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

from fastspliceai.cache import VariantCache, fingerprint
from fastspliceai.server import serve

ROW = {"#CHROM": "chr9", "POS": 128504906, "REF": "T", "ALT": "G",
       "SYMBOL": "GLE1", "DS_DL": 0.99, "Consequence": "Partial exon deletion"}


class StubEngine:
    """Records what it was asked to score, so cache hits are observable."""

    def __init__(self):
        self.calls = []
        self.exons = pd.DataFrame({"name": ["NM_1", "NM_1", "NM_2"]})
        self.distance, self.mask = 500, 0

    def score(self, variants, **kw):
        self.calls.append(list(variants))
        return pd.DataFrame([dict(ROW) for _ in variants])


@pytest.fixture
def server(tmp_path):
    engine = StubEngine()
    cache = VariantCache(tmp_path / "c.db",
                         fingerprint(annotation=None, transcripts=None, models=None,
                                     distance=500, mask=0))
    httpd = serve(engine, host="127.0.0.1", port=0, cache=cache)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = "http://127.0.0.1:{}".format(httpd.server_address[1])
    yield url, engine, cache
    httpd.shutdown(); httpd.server_close(); cache.close()


def _post(url, payload):
    req = urllib.request.Request(url + "/score", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_health(server):
    url, _, _ = server
    with urllib.request.urlopen(url + "/health", timeout=10) as r:
        body = json.loads(r.read())
    assert body["status"] == "ok"
    assert body["transcripts"] == 2 and body["distance"] == 500
    assert body["cache"]["rows"] == 0


def test_score_and_then_hit_cache(server):
    url, engine, cache = server
    first = _post(url, {"variants": ["chr9-128504906-T-G"]})
    assert first["rows"] == 1 and first["scored"] == 1 and first["cached"] == 0
    assert engine.calls == [["chr9-128504906-T-G"]]

    second = _post(url, {"variants": ["chr9-128504906-T-G"]})
    assert second["cached"] == 1 and second["scored"] == 0
    assert engine.calls == [["chr9-128504906-T-G"]], "engine was called again on a hit"
    assert second["results"][0]["DS_DL"] == 0.99


def test_partial_cache_hit_only_scores_the_rest(server):
    url, engine, _ = server
    _post(url, {"variants": ["chr9-128504906-T-G"]})
    engine.calls.clear()
    out = _post(url, {"variants": ["chr9-128504906-T-G", "chr1-100-A-T"]})
    assert out["cached"] == 1 and out["scored"] == 1
    assert engine.calls == [["chr1-100-A-T"]]


def test_single_variant_as_string(server):
    url, _, _ = server
    assert _post(url, {"variants": "chr9-128504906-T-G"})["rows"] == 1


@pytest.mark.parametrize("payload,code", [
    ({}, 400),                              # no "variants"
    ({"variants": []}, 400),                # empty
    ({"variants": ["x"] * 1001}, 400),      # over the cap
])
def test_bad_requests_are_rejected(server, payload, code):
    url, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(url, payload)
    assert exc.value.code == code


def test_unknown_paths_404(server):
    url, _, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(url + "/nope", timeout=10)
    assert exc.value.code == 404


def test_stale_cache_is_not_served(server, tmp_path):
    """The guarantee that matters: a different -D must not reuse these rows."""
    url, _, _ = server
    _post(url, {"variants": ["chr9-128504906-T-G"]})
    other = VariantCache(tmp_path / "c.db",
                         fingerprint(annotation=None, transcripts=None, models=None,
                                     distance=50, mask=0))
    assert other.get("chr9-128504906-T-G") is None
    assert other.stale_rows() >= 1
    other.close()
