"""Head-to-head against stock SpliceAI: exactness and throughput.

Stock SpliceAI needs TensorFlow, so it is run out of a separate interpreter
(--stock-python) against a checkout of the upstream package (--stock-path).
Both tools are driven through their own CLIs so the comparison includes VCF
parsing and annotation lookup, not just inference.

  python benchmarks/compare.py --vcf big.vcf -n 200 \
      --stock-python /path/to/tfenv/bin/python --stock-path /path/to/stock
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[0] / "src"


def head(vcf, n, out):
    kept = 0
    with open(vcf) as fh, open(out, "w") as w:
        for line in fh:
            if line.startswith("#"):
                w.write(line)
            elif kept < n:
                w.write(line)
                kept += 1
    return kept


def info_field(path):
    """{(chrom,pos,ref,alt): SpliceAI INFO string} for every annotated record."""
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            tag = ""
            for kv in f[7].split(";"):
                if kv.startswith("SpliceAI="):
                    tag = kv[len("SpliceAI="):]
            out[(f[0], f[1], f[3], f[4])] = tag
    return out


def timed(cmd, env=None, cwd=None):
    t = time.perf_counter()
    p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    dt = time.perf_counter() - t
    if p.returncode != 0:
        sys.exit("command failed: {}\n{}".format(" ".join(map(str, cmd)), p.stderr[-2000:]))
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("-A", default="grch37")
    ap.add_argument("-D", type=int, default=50)
    ap.add_argument("-M", type=int, default=0)
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("-j", type=int, default=1)
    ap.add_argument("--models", default=str(HERE.parents[2] / "spliceai" / "models"))
    ap.add_argument("--stock-python")
    ap.add_argument("--stock-path")
    ap.add_argument("--workdir", default=".")
    a = ap.parse_args()

    wd = Path(a.workdir)
    sub = wd / "cmp_input.vcf"
    n = head(a.vcf, a.n, sub)
    fast_out, stock_out = wd / "cmp_fast.vcf", wd / "cmp_stock.vcf"

    env = dict(os.environ, PYTHONPATH=str(SRC))
    t_fast = timed(
        [sys.executable, "-m", "fastspliceai.cli", "-I", str(sub), "-O", str(fast_out),
         "-R", a.fasta, "-A", a.A, "-D", str(a.D), "-M", str(a.M),
         "-m", a.models, "-j", str(a.j)], env=env)
    print("FastSpliceAI : {:8.1f} s   {:7.2f} variants/s   (-j {})".format(t_fast, n / t_fast, a.j))

    if not a.stock_python:
        return
    senv = dict(os.environ, TF_USE_LEGACY_KERAS="1", TF_CPP_MIN_LOG_LEVEL="3")
    t_stock = timed(
        [a.stock_python, "-m", "spliceai", "-I", str(sub), "-O", str(stock_out),
         "-R", a.fasta, "-A", a.A, "-D", str(a.D), "-M", str(a.M)],
        env=senv, cwd=a.stock_path)
    print("stock SpliceAI: {:8.1f} s   {:7.2f} variants/s".format(t_stock, n / t_stock))
    print("speedup       : {:8.2f}x".format(t_stock / t_fast))

    f, s = info_field(fast_out), info_field(stock_out)
    keys = sorted(set(f) | set(s))
    diffs = [k for k in keys if f.get(k) != s.get(k)]
    print("\n{} records compared, {} identical, {} differing".format(
        len(keys), len(keys) - len(diffs), len(diffs)))
    for k in diffs[:10]:
        print("  {}:{} {}>{}\n    stock {}\n    fast  {}".format(
            k[0], k[1], k[2], k[3], s.get(k), f.get(k)))
    sys.exit(1 if diffs else 0)


if __name__ == "__main__":
    main()
