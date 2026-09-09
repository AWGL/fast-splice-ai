"""Where FastSpliceAI's speed comes from, measured on the same weights.

Laptops throttle, so every figure is a best-of-N; the minimum is the most
stable estimator of achievable speed.
"""
import argparse, sys, time
from collections import namedtuple
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fastspliceai import Annotator, Ensemble, get_delta_scores          # noqa: E402
from fastspliceai.engine import CONTEXT                                 # noqa: E402
from fastspliceai.score import _RefCache                                # noqa: E402

Variant = namedtuple("Variant", "chrom pos ref alts")


def best(fn, n=4):
    fn()
    out = []
    for _ in range(n):
        t = time.perf_counter(); fn(); out.append(time.perf_counter() - t)
    return min(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../spliceai/models")
    ap.add_argument("--fasta", default="../tests/data/test.fa")
    ap.add_argument("-D", type=int, default=50)
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("-t", "--threads", type=int, default=0)
    a = ap.parse_args()

    ens = Ensemble(a.models, threads=a.threads or None)
    ann = Annotator(a.fasta, "grch37")
    wid = CONTEXT + 2 * a.D + 1
    rng = np.random.default_rng(0)
    x = np.zeros((1, 4, wid), np.float32)
    x[0, rng.integers(0, 4, wid), np.arange(wid)] = 1

    one = best(lambda: ens.sessions[0].run(None, {"seq": x}), a.n)
    print("window {} nt, {} threads, best of {}\n".format(wid, ens.sessions[0].get_session_options().intra_op_num_threads or "all", a.n))
    print("  one model, one sequence          : {:7.1f} ms".format(one * 1e3))
    print("  -> a biallelic SNV needs 10 such passes (5 models x ref/alt)\n")

    print("  scheduling, per site:")
    for n_alt, label in ((1, "biallelic SNV "), (3, "tri-allelic   ")):
        rec = Variant("10", 94077, "A", ["C", "G", "T"][:n_alt])
        t = best(lambda r=rec: get_delta_scores(r, ann, ens, a.D, 0, ref_cache=_RefCache(ens)), 2)
        upstream_passes = 10 * n_alt          # upstream re-runs ref for every allele
        fast_passes = 5 + 5 * n_alt           # we run ref once
        print("    {} : {:7.0f} ms   {:2d} passes vs {:2d} upstream  ({:.2f}x)".format(
            label, t * 1e3, fast_passes, upstream_passes, upstream_passes / fast_passes))

    print("\n  tiled scanning (whole-region mode):")
    L = CONTEXT + 100_000
    big = np.zeros((4, L), np.float32)
    big[rng.integers(0, 4, L), np.arange(L)] = 1
    t0 = time.perf_counter(); ens.tile(big); t_tile = time.perf_counter() - t0
    tiled = t_tile / 100_000
    windowed = one * 5 / (2 * a.D + 1)     # 5 models over one window -> 2D+1 positions
    print("    per scored base, windowed      : {:9.2f} us".format(windowed * 1e6))
    print("    per scored base, 100kb tile    : {:9.2f} us   ({:.0f}x)".format(
        tiled * 1e6, windowed / tiled))


if __name__ == "__main__":
    main()
