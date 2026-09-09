"""CPU detection must respect the allocation, not the machine.

os.cpu_count() reports the host. On a shared server or in a CPU-limited
container, sizing thread pools from it oversubscribes badly - slower for us and
antisocial to everyone else on the node.
"""
import os

import pytest

from fastspliceai.engine import available_cpus


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("FASTSPLICEAI_CPUS", "SLURM_CPUS_PER_TASK", "NSLOTS", "PBS_NP",
                "OMP_NUM_THREADS"):
        monkeypatch.delenv(var, raising=False)


def test_never_zero_or_negative(clean_env):
    assert available_cpus() >= 1


def test_does_not_exceed_the_machine(clean_env):
    assert available_cpus() <= (os.cpu_count() or 1)


@pytest.mark.parametrize("var", ["FASTSPLICEAI_CPUS", "SLURM_CPUS_PER_TASK",
                                 "NSLOTS", "PBS_NP", "OMP_NUM_THREADS"])
def test_scheduler_variables_cap_it(clean_env, monkeypatch, var):
    monkeypatch.setenv(var, "2")
    assert available_cpus() == 2, "{} was ignored".format(var)


def test_takes_the_minimum_not_the_last(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert available_cpus() == 3


def test_garbage_values_are_ignored(clean_env, monkeypatch):
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.setenv("OMP_NUM_THREADS", "0")
    assert available_cpus() >= 1


def test_engine_thread_budget_follows_it(clean_env, monkeypatch):
    """Two CPUs must mean two threads, not five workers of one each."""
    from pathlib import Path

    from fastspliceai.engine import Ensemble

    models = Path(os.environ.get("SPLICEAI_MODELS",
                                 Path(__file__).resolve().parents[2] / "spliceai" / "models"))
    if not models.is_dir():
        pytest.skip("model weights not available")
    monkeypatch.setenv("FASTSPLICEAI_CPUS", "2")
    e = Ensemble(str(models))
    per = e.sessions[0].get_session_options().intra_op_num_threads
    assert per * e._workers <= 2, "{} workers x {} threads exceeds the 2-CPU budget".format(
        e._workers, per)
