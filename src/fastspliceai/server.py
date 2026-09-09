"""A small HTTP API over a warm engine.

Start-up costs about 3 seconds - imports, five models, the transcript tables -
against ~0.5 s to score one variant. Run it once and query it, and an
interactive lookup goes from ~3 s to ~0.5 s, or to nothing at all on a cache hit.

    fastspliceai-serve -R hg38.fa -A ann.txt -t gp.txt -m /path/to/models \\
        --cache ~/.cache/fastspliceai/scores.db

    curl -s localhost:8722/health
    curl -s localhost:8722/score -d '{"variants": ["chr9-128504906-T-G"]}'

By default the response carries the delta scores and positions, the REF/ALT
probabilities behind them, and the predicted consequence. Add
{"fields": "all"} for every column, or {"fields": [...]} to choose.

{"all_scores": true} adds ALL_NON_ZERO_SCORES: every position in the window
carrying a signal, not just the four maxima. Useful for a single variant you
are interpreting; skipped by the cache, so do not use it for bulk work.

SECURITY: there is no authentication, no TLS and no audit log, and it binds
127.0.0.1 for that reason. It is a local accelerator, not something to expose.
Put it behind infrastructure you already trust before it sees patient data.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger("fastspliceai.server")
MAX_BODY = 1 << 20          # 1 MB of variants is ~20k of them
MAX_VARIANTS = 1000

# What a caller actually wants back. classify() produces ~110 columns, most of
# them intermediate values; returning all of them by default buries the scores.
# Pass {"fields": "all"} for everything, or a list to choose.
DEFAULT_FIELDS = [
    "#CHROM", "POS", "REF", "ALT", "SYMBOL", "Used_RefSeq_Transcript",
    "EXON", "INTRON", "strand",
    # delta scores and the positions they occur at
    "DS_AG", "DS_AL", "DS_DG", "DS_DL",
    "DP_AG", "DP_AL", "DP_DG", "DP_DL",
    # the reference/alternate probabilities each delta is the difference of
    "DS_AG_REF", "DS_AG_ALT", "DS_AL_REF", "DS_AL_ALT",
    "DS_DG_REF", "DS_DG_ALT", "DS_DL_REF", "DS_DL_ALT",
    # interpretation
    "Consequence", "Event_confidence", "Confidence_note",
    "Predicted_LOF", "LOF_note", "Predicted_outcomes",
    "Protein_consequence", "PTC_codon", "NMD_predicted",
]


class _Handler(BaseHTTPRequestHandler):
    engine = None            # set in serve()
    cache = None
    lock = None
    queue = None

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):          # route through logging, not stderr
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, code, payload):
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            return self._send(404, {"error": "not found"})
        c = self.cache
        self._send(200, {
            "status": "ok",
            "transcripts": int(self.engine.exons["name"].nunique()) if self.engine.exons is not None else None,
            "distance": self.engine.distance,
            "mask": self.engine.mask,
            "sai10k": self.engine.exons is not None,
            "cache": None if c is None or not c.enabled else {
                "rows": len(c), "hits": c.hits, "misses": c.misses,
                "fingerprint": c.fingerprint, "stale_rows": c.stale_rows()},
        })

    def do_POST(self):
        if self.path.rstrip("/") != "/score":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._send(400, {"error": "bad Content-Length"})
        if length <= 0 or length > MAX_BODY:
            return self._send(413, {"error": "body must be 1..{} bytes".format(MAX_BODY)})
        try:
            req = json.loads(self.rfile.read(length))
            all_scores = bool(req.get("all_scores", False))
            fields = req.get("fields", DEFAULT_FIELDS)
            if all_scores and isinstance(fields, list):
                fields = fields + ["ALL_NON_ZERO_SCORES", "SCORES_FOR_INSERTED_BASES"]
            if fields == "all":
                fields = None
            elif not isinstance(fields, list):
                raise ValueError('"fields" must be a list or "all"')
            variants = req["variants"]
            if isinstance(variants, str):
                variants = [variants]
            if not variants or len(variants) > MAX_VARIANTS:
                raise ValueError("1..{} variants".format(MAX_VARIANTS))
        except Exception as exc:
            return self._send(400, {"error": str(exc)})

        t0 = time.perf_counter()
        rows, todo, cached = [], [], 0
        for v in variants:
            # a cached row was stored without the profile, so skip the cache
            # when it is asked for rather than silently returning less
            hit = (self.cache.get(str(v))
                   if self.cache is not None and not all_scores else None)
            if hit is None:
                todo.append(str(v))
            else:
                cached += 1
                rows.extend(hit)

        if todo:
            # One engine, one request at a time: ORT already uses every core, so
            # concurrent requests would only fight each other. The semaphore
            # bounds how many may queue, so a burst is refused rather than
            # accumulating unbounded work on a shared host.
            if self.queue is not None and not self.queue.acquire(blocking=False):
                return self._send(503, {"error": "busy, too many queued requests"})
            try:
                with self.lock:
                    df = self.engine.score(todo, include_unscored=False,
                                           all_scores=all_scores)
            finally:
                if self.queue is not None:
                    self.queue.release()
            scored = {}
            if not df.empty:
                for rec in df.to_dict("records"):
                    key = "{}-{}-{}-{}".format(rec["#CHROM"], rec["POS"],
                                               rec["REF"], rec["ALT"])
                    scored.setdefault(key, []).append(rec)
                rows.extend(df.to_dict("records"))
            if self.cache is not None:
                # cache the empty list too, so unscoreable variants stay cheap
                self.cache.put_many([(v, scored.get(_norm(v), [])) for v in todo])

        if fields is not None:
            rows = [{k: r[k] for k in fields if k in r} for r in rows]
        self._send(200, {"variants": len(variants), "rows": len(rows),
                         "cached": cached, "scored": len(todo),
                         "seconds": round(time.perf_counter() - t0, 3),
                         "results": rows})


def _norm(v):
    return str(v).replace(":", "-")


def serve(engine, host="127.0.0.1", port=8722, cache=None, max_queue=32):
    import threading

    _Handler.engine = engine
    _Handler.cache = cache
    _Handler.lock = threading.Lock()
    _Handler.queue = threading.Semaphore(max_queue)
    httpd = ThreadingHTTPServer((host, port), _Handler)
    log.info("listening on http://%s:%d  (POST /score, GET /health)", host, port)
    return httpd


def main(argv=None):
    from .api import SpliceAI
    from .cache import VariantCache, fingerprint

    p = argparse.ArgumentParser(prog="fastspliceai-serve", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-R", "--reference", required=True)
    p.add_argument("-A", "--annotation", required=True)
    p.add_argument("-t", "--transcripts", help="genePred table; enables SAI-10k-calc")
    p.add_argument("-m", "--models", required=True)
    p.add_argument("-D", "--distance", type=int, default=500,
                   help="window each side of the variant (default: 500)")
    p.add_argument("--threads", type=int, default=None,
                   help="cap inference threads. Without this the engine uses the "
                        "whole CPU allocation, which is wrong when sharing a host "
                        "with other services. Also settable as FASTSPLICEAI_CPUS.")
    p.add_argument("--max-queue", type=int, default=32,
                   help="reject requests once this many are waiting (default: 32), "
                        "so a burst returns 503 instead of piling up")
    p.add_argument("-M", "--mask", type=int, choices=(0, 1), default=0)
    p.add_argument("--host", default="127.0.0.1",
                   help="default 127.0.0.1; there is no auth, see the module docstring")
    p.add_argument("--port", type=int, default=8722)
    p.add_argument("--cache", help="SQLite file for scored variants")
    p.add_argument("--purge-stale", action="store_true",
                   help="drop cache rows from a previous annotation or model set")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    engine = SpliceAI(reference=a.reference, annotation=a.annotation,
                      transcripts=a.transcripts, models=a.models,
                      distance=a.distance, mask=a.mask, threads=a.threads)
    cache = None
    if a.cache:
        fp = fingerprint(annotation=a.annotation, transcripts=a.transcripts,
                         models=a.models, distance=a.distance, mask=a.mask)
        cache = VariantCache(a.cache, fp)
        stale = cache.stale_rows()
        if stale and a.purge_stale:
            log.info("cache: purged %d stale rows", cache.purge_stale())
        elif stale:
            log.info("cache: %d rows from a different annotation/model set are "
                     "present and will be ignored (--purge-stale to remove)", stale)
    if a.host != "127.0.0.1":
        log.warning("binding %s: this server has no authentication or TLS", a.host)
    httpd = serve(engine, a.host, a.port, cache, max_queue=a.max_queue)

    # Containers are stopped with SIGTERM. http.server does not handle it, so
    # podman waited 10s and then SIGKILLed us, which risks losing cache writes.
    def _stop(signum, _frame):
        log.info("received %s, shutting down", signal.Signals(signum).name)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:                       # pragma: no cover
        pass
    finally:
        httpd.server_close()
        if cache is not None:
            cache.close()
        log.info("stopped")


if __name__ == "__main__":
    main()
