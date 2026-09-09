# SAI-10k-calc

A pure-Python port of [SAI-10k-calc](https://github.com/adavi4/SAI-10k-calc)
(Canson, Davidson *et al.*, Bioinformatics 2023; Canson, Parsons *et al.*,
Genome Research 2025), which classifies SpliceAI output into splicing
aberration types. No R, no Bioconductor, no network.

```sh
fastspliceai -I in.vcf -O scored.vcf -R hg19.fa -A ann.txt -D 500 --raw-scores
fastspliceai-sai10k -I scored.vcf -t refseq_genepred.txt -O calls.tsv
```

Verified against the R implementation's own example output: **22/22 comparable
columns match exactly across all 10 rows**, including exon numbering on both
strands, intron assignment, frameshift flags and predicted event sizes.

`--raw-scores` matters here: SAI-10k-calc's improved partial-exon-deletion and
partial-intron-retention logic needs the REF/ALT scores, not just the deltas.

The R original makes 22 `rowwise()` passes over the table, which is
row-at-a-time interpretation. This port is vectorised and classifies about
5,500 variant-annotations per second.

## Events predicted

| event | detail columns |
|---|---|
| Exon skipping | `Lost_exons`, `Lost_exons_combined_size`, `Exon_skipping_frameshift` |
| Pseudoexon activation | `Pseudoexon_intron`, `_start`, `_end`, `Gained_exon_size`, `_frameshift` |
| Intron retention | `Retained_intron`, `Retained_intron_size`, `_frameshift` |
| Partial exon deletion | `bp_5prime`, `bp_3prime` |
| Partial intron retention | `bp_5prime`, `bp_3prime` |
| Increased exon inclusion | `Exon_with_increased_inclusion` |

`Consequence` summarises whichever fired as one line, in the style of
SpliceAI-lookup:

```
Exon 11 skipping (146bp) - frameshift + premature stop at codon 551, NMD predicted
```

Increased exon inclusion has never been observed firing here — but it does not
fire on any of the R implementation's own ten example variants either, so this
is a rare event rather than a gap in the port.

## Protein consequence

The `*_aaseq` prediction is ported. It reconstructs the altered transcript,
translates from the annotated start codon and reports the divergence in the R
tool's format — three wild-type residues of context, then the altered sequence
to the first stop:

```
Protein_consequence   LGT[QLGRCVRHLW*]
PTC_codon             1813
NMD_predicted         NO
```

Verified against the R implementation on every one of its informative example
rows (4/4), including its interpretive outputs `does not affect coding region`
and `native start or stop is lost`.

Two things go beyond the R tool: **`PTC_codon`**, where the premature stop
lands, and **`NMD_predicted`**, applying the 50-nucleotide rule — a stop more
than 50 nt upstream of the final exon-exon junction triggers nonsense-mediated
decay, downstream of it escapes and yields a truncated protein.

That distinction matters. `- in-frame` on its own is misleading: NF1
`c.288+2025T>G` activates an *in-frame* 108 bp pseudoexon whose first codon is
a stop, making it a complete null.

**`NMD_predicted` is unvalidated.** The R tool produces no NMD calls, so unlike
everything else here there is no reference implementation to check against. The
`aaseq` beneath it matches theirs 4/4 and `PTC_codon` follows from a validated
translation, but the decay call itself is a heuristic with known exceptions
(AUG-proximal stops, long 3'UTRs, gene-specific escape). Treat it as triage,
not a finding.

## Confidence

Every call carries `Event_confidence` (`high` / `moderate` / `low`) and
`Confidence_note` giving the limiting delta score. A call that cannot be
anchored to a named intron or exon is forced to `low` however high its scores —
that catches the case where SAI-10k's permissive `DS_ALDL_MIN_T` floor lets a
0.05 acceptor-loss score support an intron-retention call.

The default for `DS_ALDL_MIN_T` here is **0.05**, where SAI-10k-calc uses 0.02.
Pass `Thresholds(DS_ALDL_MIN_T=0.02)` to reproduce the R tool exactly, or
`Thresholds.strict()` for tighter floors throughout.

---

*Part of the [fast-splice-ai](../README.md) documentation.*
