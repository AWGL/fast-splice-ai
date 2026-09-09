"""Invariants that must hold however the work is scheduled or computed.

Written after a regression that no existing test caught: an INT8 path was
calibrated on random sequence, every strong call collapsed to noise, and the
whole suite stayed green because nothing asserted that a known-pathogenic
variant still scores highly. INT8 has since been removed, but the gap it
exposed was real.

The lesson generalised: assert on *behaviour that must not change*, not only on
one golden file. A variant whose biology is not in doubt is a better regression
test than any fixture.
"""
import os
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = Path(os.environ.get("SPLICEAI_MODELS", ROOT / "spliceai" / "models"))
REFERENCE = Path("/data/resources/reference_genomes/"
                 "GRCh38_full_analysis_set_plus_decoy_hla_masked.fa")
ANN = HERE.parents[0] / "data" / "mane_v1.5" / "annotation.txt.gz"
GP = HERE.parents[0] / "data" / "mane_v1.5" / "genepred.txt.gz"

needs_grch38 = pytest.mark.skipif(
    not (REFERENCE.exists() and ANN.exists() and MODELS.is_dir()),
    reason="needs a GRCh38 reference, the MANE tables and the model weights")

# variants whose effect is not in doubt: canonical GT donor disruptions and
# published pseudoexon-activating deep intronic variants. If a change makes any
# of these stop scoring highly, the change is wrong.
STRONG = [
    ("chr9-128504906-T-G", "GLE1 donor +2", 0.9),
    ("chr9-21994137-A-C", "CDKN2A p14ARF donor +2", 0.9),
    ("chr8-140300616-T-G", "TRPS1 exon 11 skipping", 0.7),
    ("chr17-31161118-T-G", "NF1 c.288+2025T>G pseudoexon", 0.7),
    ("chr1-215891198-T-C", "USH2A c.7595-2144A>G pseudoexon", 0.5),
]
SCORES = ["DS_AG", "DS_AL", "DS_DG", "DS_DL"]


def _engine(**kw):
    from fastspliceai import SpliceAI
    kw.setdefault("distance", 500)
    return SpliceAI(reference=str(REFERENCE), annotation=str(ANN),
                    transcripts=str(GP), models=str(MODELS), **kw)


@pytest.fixture(scope="module")
def fp32():
    return _engine()


# ---------------------------------------------------------------- signal
@needs_grch38
@pytest.mark.parametrize("variant,label,floor", STRONG)
def test_known_pathogenic_variants_score_highly(variant, label, floor, fp32):
    df = fp32.score([variant])
    assert not df.empty, "{} produced no rows".format(label)
    assert df[SCORES].to_numpy().max() >= floor, (
        "{}: max delta score {:.2f} < {}".format(label, df[SCORES].to_numpy().max(), floor))


@needs_grch38
def test_raw_scores_reconstruct_deltas(fp32):
    """DS_XX must equal its own ALT minus REF pair, or the two disagree."""
    df = fp32.score([v for v, _, _ in STRONG], raw_scores=True)
    for base in ("DS_AG", "DS_AL", "DS_DG", "DS_DL"):
        ref, alt = df[base + "_REF"], df[base + "_ALT"]
        recomputed = (alt - ref) if base.endswith("G") else (ref - alt)
        assert np.allclose(recomputed, df[base], atol=0.011), base


# ------------------------------------------------------------- scheduling
@needs_grch38
@pytest.mark.slow
def test_process_count_does_not_change_results(tmp_path):
    """Sharding across -j workers must not alter a single score."""
    import subprocess
    import sys

    vcf = tmp_path / "in.vcf"
    with open(vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n##contig=<ID=chr9,length=138394717>\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for v, _, _ in STRONG:
            c, pos, ref, alt = v.split("-")
            if c == "chr9":
                f.write("{}\t{}\t.\t{}\t{}\t.\t.\t.\n".format(c, pos, ref, alt))

    # the child needs to find the package when running from a source checkout
    env = dict(os.environ)
    src = str(ROOT / "fastspliceai" / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    outs = []
    for j in (1, 3):
        out = tmp_path / "out{}.vcf".format(j)
        subprocess.run([sys.executable, "-m", "fastspliceai.cli", "-I", str(vcf),
                        "-O", str(out), "-R", str(REFERENCE), "-A", str(ANN),
                        "-m", str(MODELS), "-D", "500", "-j", str(j)],
                       check=True, capture_output=True, env=env)
        outs.append([l for l in open(out) if not l.startswith("#")])
    assert outs[0] == outs[1], "-j changed the output"


