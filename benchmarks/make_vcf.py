"""Generate test VCFs over the bundled test region.

Two modes:

  --mode random      N random variants inside TUBB8 (bulk throughput work)
  --mode stratified  a deliberately diverse set that hits the code paths most
                     likely to diverge: annotated splice-site boundaries, deep
                     exonic/intronic positions, every indel length, multi-allelic
                     sites, and each of the "skip this record" branches.

The shipped test FASTA covers chr10:1-110000 with real sequence from 60001.
TUBB8 (chr10:92893-95529, minus strand) is the only transcript with full 10 kb
context on both sides, so genuine multi-gene and plus-strand coverage needs a
real reference genome -- see --fasta.
"""
import argparse
import random
from pathlib import Path

from pyfaidx import Fasta

CHROM, TX_START, TX_END = "chr10", 92893, 95529
# annotated acceptor/donor boundaries of TUBB8 in the bundled grch37 table
BOUNDARIES = [92893, 94054, 94555, 94665, 94744, 94852, 95122, 95529]
HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr10,length=110000>\n"
    "##ALT=<ID=DEL,Description=\"Deletion\">\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def _seq(fasta):
    return Fasta(str(fasta), rebuild=False)[CHROM][:].seq.upper()


def _snv(seq, pos, rng):
    ref = seq[pos - 1]
    return (pos, ref, rng.choice([b for b in "ACGT" if b != ref])) if ref in "ACGT" else None


SYMBOLIC_LABELS = {"symbolic", "star_allele"}


def stratified(fasta, seed=0, exclude_symbolic=False):
    """A small, deliberately varied set. Returns [(pos, ref, alt, label)]."""
    seq = _seq(fasta)
    rng = random.Random(seed)
    rows, seen = [], set()

    def add(item, label):
        if item and (item[0], item[1], item[2]) not in seen:
            seen.add((item[0], item[1], item[2]))
            rows.append((item[0], item[1], item[2], label))

    # 1. at and around every annotated splice-site boundary -> exercises -M 1
    for b in BOUNDARIES:
        for off in (-20, -5, -2, -1, 0, 1, 2, 5, 20):
            p = b + off
            if TX_START <= p <= TX_END:
                add(_snv(seq, p, rng), "boundary{:+d}".format(off))

    # 2. deep exonic and deep intronic SNVs
    for p in (93500, 93900, 94300, 94950, 95300):
        add(_snv(seq, p, rng), "deep")

    # 3. every indel length we support
    for p, n in ((93100, 1), (93200, 2), (93300, 3)):
        ref = seq[p - 1]
        if ref in "ACGT":
            add((p, ref, ref + "".join(rng.choice("ACGT") for _ in range(n))), "ins{}".format(n))
    for p, n in ((93400, 2), (93600, 3), (93700, 4), (93800, 5)):
        ref = seq[p - 1 : p - 1 + n]
        if len(ref) == n and all(b in "ACGT" for b in ref):
            add((p, ref, ref[0]), "del{}".format(n - 1))

    # 4. skip paths -- both tools should decline these identically
    add((93050, seq[93049], ",".join(sorted({c for c in "ACGT" if c != seq[93049]}))), "multiallelic")
    long_ref = seq[93900 : 93900 + 150]
    if all(b in "ACGT" for b in long_ref):
        add((93901, long_ref, long_ref[0]), "del_too_long")          # > 2*D at D=50
    add((94100, seq[94099:94102], "TTT"), "mnp")                      # ref>1 and alt>1
    add((94200, seq[94199], "<DEL>"), "symbolic")
    add((94250, seq[94249], "*"), "star_allele")
    add((70000, seq[69999], "A" if seq[69999] != "A" else "C"), "outside_gene")
    add((3000, "N", "A"), "near_chrom_start")                         # no 10kb context
    wrong = "A" if seq[94400] != "A" else "C"
    add((94401, wrong, "G"), "ref_mismatch")                          # ref disagrees with fasta
    if exclude_symbolic:
        # stock SpliceAI cannot write a record carrying a symbolic ALT: it dies
        # with "Invalid BCF, the INFO tag id=N is too large" and truncates the
        # output. Excluded so the head-to-head can run to completion.
        rows = [r for r in rows if r[3] not in SYMBOLIC_LABELS]
    return rows


def random_set(fasta, n, seed=0, indel_fraction=0.25):
    seq = _seq(fasta)
    rng = random.Random(seed)
    seen, out = set(), []
    positions = list(range(TX_START, TX_END + 1))
    rng.shuffle(positions)
    i = 0
    while len(out) < n and i < 200 * len(positions):
        pos = positions[i % len(positions)]
        i += 1
        ref = seq[pos - 1]
        if ref not in "ACGT":
            continue
        roll = rng.random()
        if roll >= indel_fraction:
            alt = rng.choice([b for b in "ACGT" if b != ref])
        elif roll < indel_fraction / 2:
            alt = ref + "".join(rng.choice("ACGT") for _ in range(rng.randint(1, 3)))
        else:
            span = rng.randint(2, 5)
            ref = seq[pos - 1 : pos - 1 + span]
            if len(ref) < span or any(b not in "ACGT" for b in ref):
                continue
            alt = ref[0]
        if (pos, ref, alt) in seen:
            continue
        seen.add((pos, ref, alt))
        out.append((pos, ref, alt, "random"))
    return out


def write(path, rows):
    with open(path, "w") as f:
        f.write(HEADER)
        for pos, ref, alt, label in sorted(rows):
            f.write("{}\t{}\t{}\t{}\t{}\t.\t.\t.\n".format(CHROM, pos, label, ref, alt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("random", "stratified"), default="stratified")
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("-o", default="out.vcf")
    ap.add_argument("--fasta", default=str(Path(__file__).resolve().parents[2] / "tests/data/test.fa"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-symbolic", action="store_true",
                    help="drop <DEL>/* alleles, which stock SpliceAI cannot write")
    a = ap.parse_args()
    rows = (stratified(a.fasta, a.seed, a.exclude_symbolic) if a.mode == "stratified"
            else random_set(a.fasta, a.n, a.seed))
    write(a.o, rows)
    from collections import Counter
    c = Counter(r[3].rstrip("+-0123456789") or r[3] for r in rows)
    print("{}: {} variants".format(a.o, len(rows)))
    for k, v in sorted(c.items()):
        print("   {:20} {}".format(k, v))
