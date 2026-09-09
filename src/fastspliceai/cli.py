"""Command-line interface. Flags are a superset of SpliceAI v1.3.1's."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import namedtuple

from . import __version__
from .annotator import Annotator
from .engine import Ensemble, available_cpus
from .score import _RefCache, get_delta_scores

Variant = namedtuple("Variant", "chrom pos ref alts")
RAW_SUFFIX = (
    ", plus the reference and variant scores each delta is derived from. "
    "Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL"
    "|DS_AG_REF|DS_AG_ALT|DS_AL_REF|DS_AL_ALT|DS_DG_REF|DS_DG_ALT|DS_DL_REF|DS_DL_ALT"
)
INFO_LINE = (
    '##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
    "annotation. These include delta scores (DS) and delta positions (DP) for "
    "acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). "
    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">'
)

_WORKER = {}
# built once in the parent, inherited by forked workers copy-on-write
_SHARED = {}


def build_parser():
    p = argparse.ArgumentParser(
        prog="fastspliceai",
        description="FastSpliceAI {} - accelerated CPU inference for SpliceAI".format(__version__),
    )
    p.add_argument("-I", metavar="input", default="-", help="input VCF (default: stdin)")
    p.add_argument("-O", metavar="output", default="-", help="output VCF (default: stdout)")
    p.add_argument("-R", metavar="reference", required=True, help="reference genome FASTA")
    p.add_argument("-A", metavar="annotation", required=True,
                   help="'grch37', 'grch38', or a path to a custom annotation table")
    p.add_argument("-D", metavar="distance", type=int, default=500,
                   help="max distance between variant and gained/lost site (default: 50)")
    p.add_argument("-M", metavar="mask", type=int, choices=(0, 1), default=0,
                   help="mask annotated gains / unannotated losses (default: 0)")
    g = p.add_argument_group("performance")
    g.add_argument("-m", "--models", required=True,
                   help="directory holding spliceai1..5.h5 (required: the weights "
                        "are never discovered implicitly)")
    g.add_argument("-t", "--threads", type=int, default=0,
                   help="inference threads per process (default: all cores)")
    g.add_argument("-j", "--processes", type=int, default=1,
                   help="worker processes (default: 1). The five models already "
                        "run concurrently across all cores in one process, so "
                        "extra workers usually contend; raise this only if you "
                        "have many more cores than ~15")
    p.add_argument("--raw-scores", action="store_true",
                   help="also emit the REF/ALT scores each delta is derived from "
                        "(the input format SAI-10k-calc's REF/ALT mode expects)")
    g.add_argument("--chunk-size", type=int, default=500,
                   help="records held in memory at once (default: 500)")
    g.add_argument("--cache-size", type=int, default=256,
                   help="reference windows kept in memory (default: 256)")
    return p


def _init_worker(models, threads, ref, annotation):
    # ONNX sessions are built here, in the child: they must not cross a fork.
    _WORKER["ens"] = Ensemble(models, threads=threads or None)
    _WORKER["ann"] = Annotator(ref, _SHARED.get("tables", annotation))


def _score_chunk(args):
    variants, dist, mask, cache_size, raw = args
    ens, ann = _WORKER["ens"], _WORKER["ann"]
    cache = _RefCache(ens, limit=cache_size)
    return [
        get_delta_scores(v, ann, ens, dist, mask, ref_cache=cache, raw_scores=raw)
        for v in variants
    ]


def main(argv=None):
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.WARNING)
    if not os.path.isdir(args.models):
        sys.exit("error: --models is not a directory: {}".format(args.models))

    import pysam

    # hand pysam "-" directly; it opens stdin/stdout natively. Passing the
    # Python file object works only for pipes and fails on a redirected file.
    src = args.I
    try:
        vcf = pysam.VariantFile(src)
    except (IOError, ValueError) as exc:
        sys.exit("error: {}".format(exc))
    header = vcf.header
    info_line = INFO_LINE
    if args.raw_scores:
        info_line = info_line.replace(
            'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">',
            RAW_SUFFIX[2:] + '">')
    header.add_line(info_line)
    dst = args.O
    out = pysam.VariantFile(dst, mode="w", header=header)

    def chunks(iterable, n):
        buf = []
        for item in iterable:
            buf.append(item)
            if len(buf) >= n:
                yield buf
                buf = []
        if buf:
            yield buf

    def emit(records, results):
        for record, scores in zip(records, results):
            if scores:
                record.info["SpliceAI"] = scores
            # Re-map onto the output header's dictionary. Without this, records
            # carrying a symbolic ALT (<DEL>, <DUP>, ...) fail to write with
            # "Invalid BCF, the INFO tag id=N is too large".
            record.translate(out.header)
            out.write(record)

    def to_variants(records):
        return [Variant(r.chrom, r.pos, r.ref, list(r.alts) if r.alts else [])
                for r in records]

    # Stream in chunks rather than materialising the whole VCF: a whole-genome
    # input is millions of records, and callers want output as it is produced.
    chunk_size = max(1, args.chunk_size)

    if args.processes > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        per = max(1, (args.threads or available_cpus()) // args.processes)
        # Parse the transcript table once and fork, so workers inherit it rather
        # than each spending ~0.5s rebuilding the same index.
        kwargs = {}
        if "fork" in multiprocessing.get_all_start_methods():
            _SHARED["tables"] = Annotator.build_tables(args.A)
            kwargs["mp_context"] = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=args.processes,
            initializer=_init_worker,
            initargs=(args.models, per, args.R, args.A),
            **kwargs,
        ) as pool:
            for batch in chunks(vcf, chunk_size * args.processes):
                shard = max(1, len(batch) // args.processes + 1)
                jobs = [(to_variants(batch[i:i + shard]), args.D, args.M,
                         args.cache_size, args.raw_scores)
                        for i in range(0, len(batch), shard)]
                scored = [s for part in pool.map(_score_chunk, jobs) for s in part]
                emit(batch, scored)
    else:
        _init_worker(args.models, args.threads, args.R, args.A)
        for batch in chunks(vcf, chunk_size):
            emit(batch, _score_chunk((to_variants(batch), args.D, args.M,
                                      args.cache_size, args.raw_scores)))

    vcf.close()
    out.close()


if __name__ == "__main__":
    main()
