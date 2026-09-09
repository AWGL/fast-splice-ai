"""fastspliceai-sai10k - post-process SpliceAI output into aberration calls."""
from __future__ import annotations

import argparse
import sys

from .classify import REFALT_FIELDS, Thresholds, classify, parse_spliceai_vcf
from .exons import build_exon_table, load_genepred

OUTPUT_COLS = [
    "#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "ALLELE", "SYMBOL",
    "DS_AG", "DS_AL", "DS_DG", "DS_DL", "DP_AG", "DP_AL", "DP_DG", "DP_DL",
    "Used_RefSeq_Transcript", "strand", "Any_splicing_aberration", "Consequence",
    "bp_5prime", "bp_3prime",
    "Partial_intron_retention", "Partial_exon_deletion",
    "Gained_exon_size", "Pseudoexon_activation", "Pseudoexon_start",
    "Pseudoexon_end", "Pseudoexon_frameshift", "Pseudoexon_intron",
    "Increased_exon_inclusion", "Exon_with_increased_inclusion",
    "Exon_skipping", "Lost_exons", "Exon_skipping_frameshift",
    "Retained_intron_size", "Intron_retention", "Retained_intron",
    "Intron_retention_frameshift",
]


def main(argv=None):
    t = Thresholds()
    p = argparse.ArgumentParser(
        prog="fastspliceai-sai10k",
        description="SAI-10k-calc splicing-aberration classification (pure Python, offline).",
    )
    p.add_argument("-I", "--input", required=True, help="SpliceAI-annotated VCF")
    p.add_argument("-O", "--output", default="-", help="output TSV (default: stdout)")
    p.add_argument("-t", "--transcripts", required=True,
                   help="UCSC genePred-style transcript table (ncbiRefSeq/refGene)")
    for name in ("GEX_size_MIN", "GEX_size_MAX", "AG_T", "DG_T", "AGDG_T", "DS_GL_T",
                 "DS_AGDG_MIN_T", "DS_AGDG_MAX_T", "DS_ALDL_MIN_T", "DS_ALDL_MAX_T"):
        p.add_argument("--" + name, type=float, default=getattr(t, name))
    a = p.parse_args(argv)

    thresholds = Thresholds(**{k: getattr(a, k) for k in t.__dataclass_fields__})
    variants = parse_spliceai_vcf(a.input)
    if variants.empty:
        sys.exit("error: no SpliceAI annotations found in {}".format(a.input))
    if not all(c in variants.columns for c in REFALT_FIELDS):
        print("note: input has no REF/ALT scores; rerun fastspliceai with --raw-scores "
              "for improved partial-deletion and partial-retention calls", file=sys.stderr)
    out = classify(variants, build_exon_table(load_genepred(a.transcripts)), thresholds)
    cols = [c for c in OUTPUT_COLS if c in out.columns]
    handle = sys.stdout if a.output == "-" else open(a.output, "w")
    out[cols].to_csv(handle, sep="\t", index=False, na_rep="NA")
    if handle is not sys.stdout:
        handle.close()
        print("{}: {} rows".format(a.output, len(out)), file=sys.stderr)


if __name__ == "__main__":
    main()
