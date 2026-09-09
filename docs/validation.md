# Validation

Evidence that FastSpliceAI reproduces stock SpliceAI exactly.

`fp32` output is identical to SpliceAI v1.3.1. The test suite asserts the
published upstream vectors byte-for-byte:

```
C|TUBB8|0.15|0.27|0.00|0.05|89|-23|-267|193
T|TUBB8|0.01|0.18|0.15|0.62|-2|110|-190|0
```

```sh
pytest tests -q
```

Verified three times, against stock SpliceAI v1.3.1 running on TensorFlow with the
same five `.h5` files:

| set | build | records | identical |
|---|---|---|---|
| random SNVs + indels, TUBB8 | GRCh37 | 200 | **200 / 200** |
| stratified: splice boundaries, indels, skip paths, `-M 0/1`, `-D 50/500` | GRCh37 | 246 | **246 / 246** |
| 20 genes, 14 chromosomes, both strands, indels | **GRCh38** | 280 | **280 / 280** |

The GRCh38 set is the important one: it covers plus- and minus-strand
transcripts across real loci, which a single-gene test FASTA cannot. Together
that is 726 comparisons with no differences.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
