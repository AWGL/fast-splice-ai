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


## Beyond golden files

Golden files only catch changes to the rows they contain. `tests/test_invariants.py`
adds assertions about behaviour that must hold whatever the implementation does:

* variants whose biology is not in doubt — canonical `GT` donor disruptions and
  published pseudoexon-activating deep intronic variants — must keep scoring
  above a floor;
* `DS_XX` must equal its own ALT/REF pair, so the two outputs cannot drift;
* results must not depend on how the work is sharded across processes.

This layer was added after a regression the golden files missed entirely: an
INT8 path calibrated on random sequence turned every strong call into noise
while the suite stayed green. INT8 has since been removed; the tests remain.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
