"""SQLite cache for scored variants.

Scoring a variant costs ~0.5s and the answer never changes for a fixed
(reference, transcript set, model weights, distance, mask). So it is worth
storing - but only if a stale entry can never be served.

Every key is salted with a fingerprint of those inputs. Change the annotation
table, the model files, or -D, and the fingerprint changes, so old rows simply
stop matching rather than being silently returned against the wrong transcripts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger("fastspliceai.cache")

SCHEMA = """
CREATE TABLE IF NOT EXISTS scores (
    key     TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    created REAL NOT NULL DEFAULT (julianday('now'))
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


def fingerprint(*, annotation, transcripts, models, distance, mask, extra=""):
    """Identify everything a score depends on.

    Files contribute their size and mtime rather than their contents: hashing a
    3 GB reference on every start-up is not worth it, and a rebuilt table always
    changes its mtime.
    """
    h = hashlib.sha1()
    for item in (annotation, transcripts, models):
        if item is None:
            h.update(b"none")
            continue
        p = Path(str(item))
        if p.is_dir():
            for f in sorted(p.glob("*.h5")):
                st = f.stat()
                h.update("{}:{}:{}".format(f.name, st.st_size, int(st.st_mtime)).encode())
        elif p.exists():
            st = p.stat()
            h.update("{}:{}:{}".format(p.name, st.st_size, int(st.st_mtime)).encode())
        else:
            h.update(str(item).encode())          # e.g. "grch38"
    h.update("D{}M{}{}".format(distance, mask, extra).encode())
    return h.hexdigest()[:16]


class VariantCache:
    """Keyed on (variant, fingerprint). Safe to share across threads."""

    def __init__(self, path, fingerprint, enabled=True):
        self.fingerprint = fingerprint
        self.enabled = bool(enabled) and path is not None
        self.hits = self.misses = 0
        self._lock = threading.Lock()
        self.db = None
        if not self.enabled:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('fingerprint', ?)",
                        (fingerprint,))
        self.db.commit()
        log.info("cache: %s (fingerprint %s, %d rows)", path, fingerprint, len(self))

    def _key(self, variant):
        return "{}|{}".format(self.fingerprint, variant)

    def get(self, variant):
        if not self.enabled:
            return None
        with self._lock:
            row = self.db.execute("SELECT payload FROM scores WHERE key = ?",
                                  (self._key(variant),)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, variant, rows):
        if not self.enabled:
            return
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO scores (key, payload) VALUES (?, ?)",
                            (self._key(variant), json.dumps(rows, default=str)))
            self.db.commit()

    def put_many(self, items):
        if not self.enabled:
            return
        with self._lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO scores (key, payload) VALUES (?, ?)",
                [(self._key(v), json.dumps(r, default=str)) for v, r in items])
            self.db.commit()

    def __bool__(self):
        # Without this, defining __len__ makes an empty cache falsy, so
        # `if cache:` silently skips every write and it can never fill.
        return self.enabled

    def __len__(self):
        if not self.enabled:
            return 0
        return self.db.execute("SELECT count(*) FROM scores").fetchone()[0]

    def stale_rows(self):
        """Entries left over from a previous fingerprint - never served, just dead."""
        if not self.enabled:
            return 0
        return self.db.execute(
            "SELECT count(*) FROM scores WHERE key NOT LIKE ?",
            (self.fingerprint + "|%",)).fetchone()[0]

    def purge_stale(self):
        if not self.enabled:
            return 0
        with self._lock:
            n = self.db.execute("DELETE FROM scores WHERE key NOT LIKE ?",
                                (self.fingerprint + "|%",)).rowcount
            self.db.commit()
        return n

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
