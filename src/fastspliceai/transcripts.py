"""Build annotation tables from a local GFF3/GTF. No network, no R.

This replaces SAI-10k-calc's `download_tx.R` (which pulls UCSC's ncbiRefSeq
table through rtracklayer) and lets FastSpliceAI run against any transcript set
you already have on disk - RefSeq, GENCODE, Ensembl or an in-house build.

Everything here is offline by construction: it reads a file you supply and
writes tables next to it. Nothing in FastSpliceAI's scoring path opens a socket.

Coordinate conventions
----------------------
GFF/GTF are 1-based inclusive. The SpliceAI annotation table is genePred-style:
TX_START/EXON_START are 0-based, TX_END/EXON_END are 1-based. We convert.
"""
from __future__ import annotations

import gzip
import re
from collections import defaultdict

TX_FEATURES = {"transcript", "mRNA", "primary_transcript"}
GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')


def _open(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def _attrs(field):
    """Parse the 9th GFF/GTF column into a dict, handling both dialects."""
    field = field.strip()
    if not field or field == ".":
        return {}
    if '"' in field:                                    # GTF
        return {k: v for k, v in GTF_ATTR.findall(field)}
    out = {}                                            # GFF3
    for part in field.split(";"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = v.strip()
    return out


_PREFIX = re.compile(r"^(rna-|gene-|transcript:|mRNA:|id-)")


def _norm(v):
    """Strip the decorations different GFF dialects put on the same accession."""
    return _PREFIX.sub("", v.split(":")[-1] if ":" in v and "-" not in v.split(":")[0] else v)


def _aliases(a, keys):
    out = []
    for k in keys:
        if k in a and a[k]:
            for v in a[k].split(","):
                v = v.strip()
                out.extend([v, _norm(v)])
    seen, uniq = set(), []
    for v in out:
        if v and v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _tx_id(a):
    ids = _aliases(a, ("transcript_id", "ID", "Name", "transcript_name"))
    return ids[0] if ids else None


def _name(a, tx):
    for k in ("gene_name", "gene_id", "gene"):
        if k in a:
            return a[k].split(":")[-1]
    return tx


def parse(path, feature_types=TX_FEATURES):
    """Return {tx_id: dict(name, chrom, strand, tx_start, tx_end, exons, cds)}.

    tx_start/tx_end and every coordinate stored here stay 1-based inclusive;
    conversion happens in to_spliceai_table().
    """
    tx, exons, cds, alias, loc = {}, defaultdict(list), defaultdict(list), {}, {}
    with _open(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            kind, start, end, strand, attr = f[2], int(f[3]), int(f[4]), f[6], f[8]
            a = _attrs(attr)
            if kind in feature_types:
                t = _tx_id(a)
                if t:
                    tx[t] = {
                        "name": _name(a, t), "tx_id": t, "chrom": f[0], "strand": strand,
                        "tx_start": start, "tx_end": end,
                    }
                    for v in _aliases(a, ("transcript_id", "ID", "Name", "transcript_name")):
                        alias[v] = t
            elif kind in ("exon", "CDS"):
                target = exons if kind == "exon" else cds
                for v in _aliases(a, ("transcript_id", "Parent")):
                    target[v].append((start, end))
                    # a GTF often has no transcript row at all; remember where
                    # its exons live so one can be synthesised below
                    loc.setdefault(v, (f[0], strand, _name(a, v)))
    for t, rec in tx.items():
        keys = {k for k, v in alias.items() if v == t} | {t}
        ex = sorted({e for k in keys for e in exons.get(k, [])})
        c = sorted({e for k in keys for e in cds.get(k, [])})
        rec["exons"] = ex
        rec["cds"] = (c[0][0], c[-1][1]) if c else None
    # a GTF may omit transcript rows; synthesise them from their exons
    for t, ex in exons.items():
        if t not in tx and t not in alias and ex:
            chrom, strand, name = loc.get(t, (None, None, t))
            c = sorted(set(cds.get(t, [])))
            tx[t] = {"name": name, "tx_id": t, "chrom": chrom, "strand": strand,
                     "tx_start": min(s for s, _ in ex), "tx_end": max(e for _, e in ex),
                     "exons": sorted(set(ex)), "cds": (c[0][0], c[-1][1]) if c else None}
    return {t: r for t, r in tx.items() if r["exons"] and r["chrom"]}


def to_spliceai_table(records, strip_chr=True, name_prefix="", name_field="gene"):
    """Rows for a SpliceAI -A annotation file (genePred coordinates).

    name_field="transcript" names each row by its accession (NM_000059.4), which
    is what SAI-10k-calc expects alongside name_prefix="RefSeqTx-".
    """
    rows = []
    for tx_id, r in sorted(records.items()):
        label = r.get("tx_id", tx_id) if name_field == "transcript" else r["name"]
        chrom = r["chrom"]
        if strip_chr and chrom.startswith("chr"):
            chrom = chrom[3:]
        rows.append({
            "#NAME": "{}{}".format(name_prefix, label),
            "CHROM": chrom,
            "STRAND": r["strand"],
            "TX_START": r["tx_start"] - 1,
            "TX_END": r["tx_end"],
            "EXON_START": ",".join(str(s - 1) for s, _ in r["exons"]) + ",",
            "EXON_END": ",".join(str(e) for _, e in r["exons"]) + ",",
        })
    return rows


def exon_frames(exons, strand, cds):
    """UCSC genePred exonFrames: frame of each exon's first coding base, -1 if none.

    frame = (number of coding bases in earlier exons) % 3, walking 5'->3'.
    `exons` are 1-based inclusive (start, end); `cds` is a 1-based inclusive span.
    """
    if not cds:
        return [-1] * len(exons)
    c_lo, c_hi = cds
    order = range(len(exons)) if strand == "+" else range(len(exons) - 1, -1, -1)
    frames, running = [-1] * len(exons), 0
    for i in order:
        s_, e_ = exons[i]
        lo, hi = max(s_, c_lo), min(e_, c_hi)
        if lo > hi:
            continue
        frames[i] = running % 3
        running += hi - lo + 1
    return frames


def to_genepred_table(records, keep_chr=True):
    """Rows in the UCSC genePred layout that `fastspliceai-sai10k` consumes."""
    rows = []
    for tx_id, r in sorted(records.items()):
        chrom = r["chrom"] if keep_chr or not r["chrom"].startswith("chr") else r["chrom"][3:]
        cds = r.get("cds")
        frames = exon_frames(r["exons"], r["strand"], cds)
        rows.append({
            "name": r.get("tx_id", tx_id), "chrom": chrom, "strand": r["strand"],
            "txStart": r["tx_start"] - 1, "txEnd": r["tx_end"],
            "cdsStart": (cds[0] - 1) if cds else r["tx_end"],
            "cdsEnd": cds[1] if cds else r["tx_end"],
            "exonCount": len(r["exons"]),
            "exonStarts": ",".join(str(s - 1) for s, _ in r["exons"]) + ",",
            "exonEnds": ",".join(str(e) for _, e in r["exons"]) + ",",
            "name2": r["name"],
            "exonFrames": ",".join(str(f) for f in frames) + ",",
        })
    return rows


GENEPRED_OUT = ["name", "chrom", "strand", "txStart", "txEnd", "cdsStart", "cdsEnd",
                "exonCount", "exonStarts", "exonEnds", "name2", "exonFrames"]


def write_genepred(rows, path):
    with open(path, "w") as f:
        f.write("\t".join(GENEPRED_OUT) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in GENEPRED_OUT) + "\n")
    return path


def write_table(rows, path):
    cols = ["#NAME", "CHROM", "STRAND", "TX_START", "TX_END", "EXON_START", "EXON_END"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    return path


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="fastspliceai-tx",
        description="Build a SpliceAI annotation table from a local GFF3/GTF (offline).",
    )
    ap.add_argument("gff", help="GFF3 or GTF, optionally gzipped")
    ap.add_argument("-o", "--out", required=True, help="output SpliceAI annotation table (-A)")
    ap.add_argument("--out-genepred", help="also write the genePred table that "
                    "fastspliceai-sai10k consumes (-t), from the same source")
    ap.add_argument("--genes", help="optional file of gene or transcript IDs to keep, one per line")
    ap.add_argument("--keep-chr", action="store_true", help="keep the 'chr' prefix on chromosomes")
    ap.add_argument("--name-prefix", default="", help="prefix for the NAME column, e.g. 'RefSeqTx-'")
    ap.add_argument("--name-field", choices=("gene", "transcript"), default="gene",
                    help="name rows by gene symbol (default) or transcript accession (NM_...)")
    a = ap.parse_args(argv)

    recs = parse(a.gff)
    if a.genes:
        wanted = {l.split()[0] for l in open(a.genes) if l.strip() and not l.startswith("#")}
        recs = {t: r for t, r in recs.items()
                if r["name"] in wanted or t in wanted or t.split(".")[0] in wanted}
        if not recs:
            raise SystemExit("error: no transcripts matched {}".format(a.genes))
    rows = to_spliceai_table(recs, strip_chr=not a.keep_chr, name_prefix=a.name_prefix,
                             name_field=a.name_field)
    write_table(rows, a.out)
    print("{}: {} transcripts".format(a.out, len(rows)))
    if a.out_genepred:
        write_genepred(to_genepred_table(recs), a.out_genepred)
        print("{}: {} transcripts".format(a.out_genepred, len(recs)))


if __name__ == "__main__":
    main()
