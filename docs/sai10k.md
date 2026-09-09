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

Not yet ported: the predicted amino-acid sequences (the `*_aaseq` columns),
which need the reference genome and a translation step.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
