#!/usr/bin/env python
"""Fetch the hg19 BRCA1/BRCA2 loci and N-pad them to real coordinates.

Needed only by tests/test_protein.py, which compares against SAI-10k-calc's own
hg19 example output. ~75 MB, so it is built on demand rather than committed.
"""
import json
import urllib.request
from pathlib import Path

REGIONS = {"chr13": (32889000, 32975000), "chr17": (41195000, 41278000)}
OUT = Path(__file__).resolve().parent / "data" / "hg19_brca.fa"

if __name__ == "__main__":
    with open(OUT, "w") as f:
        for chrom, (start, end) in REGIONS.items():
            url = ("https://api.genome.ucsc.edu/getData/sequence"
                   "?genome=hg19;chrom={};start={};end={}".format(chrom, start, end))
            seq = json.load(urllib.request.urlopen(url, timeout=120))["dna"].upper()
            full = "N" * start + seq
            f.write(">{}\n".format(chrom))
            for i in range(0, len(full), 60):
                f.write(full[i:i + 60] + "\n")
            print("{}:{}-{}  {} bp".format(chrom, start, end, len(seq)))
    print("wrote", OUT)
