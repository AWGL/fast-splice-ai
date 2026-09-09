# MANE v1.5 transcript tables (GRCh38)

Prebuilt so you do not need to download a GFF or wait ~30 s to parse one.

| file | used by | flag |
|---|---|---|
| `annotation.txt.gz` | `fastspliceai` | `-A` |
| `genepred.txt.gz` | `fastspliceai-sai10k` | `-t` |

Both hold the **same 19,507 transcripts** and were generated together from one
source, so they cannot disagree. `genepred.txt.gz` is a superset: it also carries
`cdsStart`, `cdsEnd`, `exonFrames` and `name2`, which SAI-10k-calc needs for
reading frame and frameshift calls but the scorer does not.

Includes **19,293 MANE Select and 74 MANE Plus Clinical** transcripts. Do not
filter Plus Clinical out - see `docs/transcripts.md`.

Chromosomes are named `chr1`-style, matching a standard GRCh38 analysis set.

## Provenance

```
source:    MANE.GRCh38.v1.5.refseq_genomic.gff.gz
           https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/
generated: fastspliceai-tx MANE.GRCh38.v1.5.refseq_genomic.gff.gz \
               -o annotation.txt --out-genepred genepred.txt \
               --name-field transcript --name-prefix RefSeqTx-
```

MANE is produced by NCBI and EMBL-EBI and is in the public domain.

To regenerate, or to move to a newer MANE release, rerun the command above and
replace both files together.
