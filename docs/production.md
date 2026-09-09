# Running in production

Nothing in the scoring path opens a network connection. Model conversion reads
your local `.h5` files; annotation tables are read from disk. The only optional
network step is fetching a transcript set, and you can skip it entirely by
supplying your own GFF3/GTF:

```sh
fastspliceai-tx refseq.gff3.gz -o refseq_tx.txt \
    --genes my_genes.txt --name-field transcript --name-prefix RefSeqTx-
fastspliceai -I in.vcf -O out.vcf -R genome.fa -A refseq_tx.txt -D 500
```

`--name-field transcript` names rows by accession (`RefSeqTx-NM_000059.4`),
which is what SAI-10k-calc expects. Both GFF3 and GTF dialects are handled,
including NCBI RefSeq (`Parent=rna-NM_...`) and GENCODE/Ensembl GTF with no
transcript rows at all. Note that NCBI GFF3 names chromosomes by accession
(`NC_000013.10`); map those to your FASTA's naming before use.

## Environments

Because TensorFlow is gone, every dependency is a plain wheel and a stock `venv`
is enough. There is no need for conda unless you already use it.

```sh
python -m venv ~/envs/fastspliceai
source ~/envs/fastspliceai/bin/activate
pip install "git+https://github.com/your-org/fastspliceai.git@v0.1.0"

fastspliceai -I in.vcf -O out.vcf -R /data/hg38.fa -A grch38 -D 500 \
    --models /data/resources/spliceai/models
```

With conda, create the environment but still install FastSpliceAI itself with
pip — it is not on conda-forge or bioconda:

```sh
conda create -n fastspliceai -c conda-forge python=3.11 pysam pandas numpy
conda activate fastspliceai
pip install "git+https://github.com/your-org/fastspliceai.git@v0.1.0"
```

## Air-gapped installs

Build a wheelhouse on a connected machine, then install from it with no network:

```sh
# connected
pip wheel "git+https://github.com/your-org/fastspliceai.git@v0.1.0" -w wheelhouse/
# air-gapped
pip install --no-index --find-links wheelhouse/ fastspliceai
```

Copy the model `.h5` files and your annotation tables across separately. First
run converts the models to ONNX locally and caches them under
`$XDG_CACHE_HOME/fastspliceai`; pre-warm that cache in the image if you want the
first scoring run to be fast.

## Reproducibility

For a pipeline whose output you have to stand behind:

* **Pin the version** with `@v0.1.0`, and pin `onnxruntime` too - it is the
  numerical engine.
* **Pin the thread count** (`-t/--threads`), and see the note below.
* **Record the model directory.** `--models` is required and never inferred, so
  the exact weights used are always visible in the command line.
* **Version your annotation tables** alongside results. Predictions are only
  meaningful against the transcript set that produced them.

**On thread counts and determinism.** Results are *not* bit-identical across
different `--threads` values, because parallel convolutions sum partial results
in a different order. Measured across 1, 2, 4 and 10 threads on the same input:

| threads | bit-identical to 1 thread | max abs difference |
|---|---|---|
| 2 | no | 4.7e-10 |
| 4 | no | 5.8e-10 |
| 10 | no | 5.8e-10 |

That is seven orders of magnitude below the 0.005 that could change a score
printed to two decimal places, so reported scores are stable in practice. It is
not a FastSpliceAI quirk - stock SpliceAI behaves the same way, for the same
reason. If you need byte-identical reruns as a matter of policy rather than
practice, fix `--threads` in your pipeline config.

Note also that ONNX Runtime's thread pool is **process-wide** and fixed by the
first `Ensemble` created. A later one asking for a different count keeps the
original and emits a `RuntimeWarning`; use separate processes if you genuinely
need different thread counts side by side.

## Memory on large VCFs

Records are streamed in chunks (`--chunk-size`, default 500) rather than read
into memory in one go, so a whole-genome VCF costs a bounded amount of RAM and
annotated records appear in the output as they are produced instead of only at
the end. With `-j`, each worker holds one shard of a chunk.

Peak RSS is roughly `181 MB x processes` for the models and annotation table,
plus the chunk itself. `-j4` at the default chunk size is comfortable on an
8 GB machine.

## Robustness

Stock SpliceAI cannot write a record carrying a symbolic ALT (`<DEL>`, `<DUP>`):
htslib rejects it with *"Invalid BCF, the INFO tag id=N is too large"* and the
run truncates silently at that record. On an 84-variant test set stock emitted
28 records and stopped. FastSpliceAI re-maps each record onto the output header
before writing and emits all 84, leaving unscoreable alleles unannotated.

---

*Part of the [FastSpliceAI](../README.md) documentation.*
