# Transcripts

How transcript choice changes predictions, where to get an annotation,
and how to keep it current.

Worth understanding before you swap annotation sets, because it is less than
people usually assume.

SpliceAI emits **one prediction per (transcript, ALT allele)**. Every transcript
row spanning the variant position produces its own row of scores.

But the network itself is **not transcript-aware** - it only ever sees a
one-hot DNA sequence. Exactly two things about a transcript reach it:

1. **Transcript boundaries.** The 10 kb window is masked to `N` outside
   `TX_START`..`TX_END`, so a transcript that starts later gives the model a
   different input sequence.
2. **Strand.** Minus-strand transcripts are reverse-complemented before scoring.

**Exon structure is not an input at all.** It is used only for `-M 1` masking
(distance to the nearest annotated boundary) and by SAI-10k-calc downstream.

Four transcripts over one locus, same variant, demonstrating each case:

| transcript | bounds | strand | exons | DS_AG | DS_AL |
|---|---|---|---|---|---|
| TX_A | 92892-95529 | - | 4 | 0.07 | 0.27 |
| TX_B | 92892-95529 | - | **3** | 0.07 | 0.27 |
| TX_C | **93500**-95529 | - | 4 | 0.06 | 0.28 |
| TX_D | 92892-95529 | **+** | 4 | 0.00 | 0.00 |

TX_A and TX_B differ in exon structure and score **identically**. TX_C moves one
boundary and the scores shift. TX_D flips strand and the prediction changes
completely.

Two consequences:

* The bundled `grch37`/`grch38` tables carry exactly one canonical transcript
  per gene (20,274 rows, 20,274 distinct names), so out of the box you get one
  prediction per *gene*. Supply a multi-transcript RefSeq table and you get one
  per *transcript* - and many will be byte-identical.
* fast-splice-ai keys its reference cache on `(window, strand)`, so identical
  transcripts cost nothing extra. In the table above, four transcripts needed
  only three reference passes.

## Where to get a GFF

All of these are GRCh38, use `chr1`-style names matching a standard analysis-set
FASTA, and parse with `fastspliceai-tx` as-is.

**MANE — recommended for clinical use.** One agreed transcript per gene, matched
between RefSeq and Ensembl. 7.9 MB.

```sh
curl -O https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.refseq_genomic.gff.gz
fastspliceai-tx MANE.GRCh38.v1.5.refseq_genomic.gff.gz \
    -o mane_annotation.txt --out-genepred mane_genepred.txt \
    --name-field transcript --name-prefix RefSeqTx-
```

(There is an `ensembl_genomic.gff.gz` alongside it if you would rather have ENST
identifiers.)

**All RefSeq transcripts**, if MANE is too narrow:

```sh
curl -O https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz
```

**GENCODE**, if you work in Ensembl identifiers:

```sh
curl -O https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz
```

Avoid the full NCBI genome GFF (`GCF_000001405.40_GRCh38.p14_genomic.gff.gz`):
it names chromosomes by accession (`NC_000001.11`), which will not join to a
`chr1`-named FASTA without a mapping step that fast-splice-ai does not yet do.

Parsing MANE takes about 30 seconds and produces 19,507 transcripts. It is a
one-off — keep the two output tables and reuse them.

## MANE Select and MANE Plus Clinical

The MANE file contains **both**, and `fastspliceai-tx` keeps both:

| tag | transcripts |
|---|---|
| `MANE Select` | 19,293 |
| `MANE Plus Clinical` | 74 |

Plus Clinical exists for the 73 genes where the Select transcript does not cover
everything clinically important — `BRCA1` is not among them, but `BRAF`,
`CDKN2A`, `CACNA1A`, `ABCC8`, `DYSF`, `CUX1` and others are.

**Do not filter Plus Clinical out.** CDKN2A is the clean illustration: MANE
Select is `NM_000077.5` (p16INK4a, chr9:21,967,752-21,974,857) and Plus Clinical
is `NM_058195.4` (p14ARF), which starts from a different first exon ~20 kb away.
The p14ARF exon 1β donor at chr9:21,994,137 lies **outside the Select transcript
altogether**:

```
 9-21994137-A-C   RefSeqTx-NM_058195.4   DS_DL 0.99      <- Plus Clinical only
9-128504906-T-G   RefSeqTx-NM_001003722.2 DS_DL 0.99     <- Select (GLE1)
```

With a Select-only annotation the CDKN2A variant produces **no output row at
all** — not a low score, no row. Both are in the worked example.

## Updating transcript models

Both tables must come from **one source, regenerated together**. They are
matched on transcript name, so if they drift apart - a RefSeq release bumps
`NM_000059.4` to `.5` in one file but not the other - transcripts silently fail
to join and variants come back unclassified rather than erroring.

One command emits both, so they cannot disagree:

```sh
fastspliceai-tx ncbiRefSeq_2026_01.gff3.gz \
    -o ann_2026_01.txt \
    --out-genepred genepred_2026_01.txt \
    --name-field transcript --name-prefix RefSeqTx- \
    --genes my_panel.txt
```

Then point both tools at that pair (`-A ann_2026_01.txt`, `-t genepred_2026_01.txt`).

Notes:

* **Exon frames are computed from the CDS** when your GFF has no genePred frame
  column, and reproduce UCSC's `exonFrames` exactly on both strands. They drive
  the frameshift calls, so a GFF without `CDS` records yields frames of -1 and
  suppresses frameshift predictions.
* **The converted ONNX models are unaffected.** They are the neural network, not
  the annotation; a transcript update never invalidates the model cache.
* **Version the tables with your results.** Aberration calls are only meaningful
  against the transcript set that produced them.
* NCBI GFF3 names chromosomes by accession (`NC_000013.10`). Map those to your
  FASTA's naming convention before use.

---

*Part of the [fast-splice-ai](../README.md) documentation.*
