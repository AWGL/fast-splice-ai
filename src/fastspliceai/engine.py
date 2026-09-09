"""ONNX Runtime inference for the five SpliceAI models."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from .convert import MODEL_NAMES, ensure_onnx

log = logging.getLogger("fastspliceai.engine")


# Leave headroom by default: this is expected to share a host with other
# services. "At most half the allocation, but never fewer than four" scales
# sensibly from a 4-core VM to a 64-core server, and is overridden by an
# explicit threads= / --threads, or by FASTSPLICEAI_CPUS.
def default_threads():
    total = available_cpus()
    capped = min(total, max(4, total // 2))
    if capped < total:
        log.info("threads: using %d of %d available (default cap; set --threads "
                 "or FASTSPLICEAI_CPUS to change)", capped, total)
    return capped


def available_cpus():
    """How many CPUs we may actually use.

    os.cpu_count() reports the machine, not the allocation. On a shared server
    or in a CPU-limited container it will happily tell you 64 when the cgroup
    grants you 2, and sizing a thread pool from it makes the job slower and a
    bad neighbour. Take the minimum of every limit we can see.
    """
    limits = {}
    try:
        limits["affinity"] = len(os.sched_getaffinity(0))    # taskset / cpuset
    except (AttributeError, OSError):
        pass
    try:                                                      # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            limits["cgroup2"] = max(1, int(float(quota) / float(period)))
    except (OSError, ValueError):
        pass
    try:                                                      # cgroup v1
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0 and period > 0:
            limits["cgroup1"] = max(1, quota // period)
    except (OSError, ValueError):
        pass
    for var in ("FASTSPLICEAI_CPUS", "SLURM_CPUS_PER_TASK", "NSLOTS", "PBS_NP",
                "OMP_NUM_THREADS"):                           # schedulers, user intent
        try:
            value = int(os.environ[var])
            if value > 0:
                limits[var] = value
        except (KeyError, ValueError):
            pass
    limits.setdefault("cpu_count", os.cpu_count() or 1)
    chosen = max(1, min(limits.values()))
    if chosen != limits.get("cpu_count"):
        log.info("cpus: using %d (limits seen: %s)", chosen,
                 ", ".join("{}={}".format(k, v) for k, v in sorted(limits.items())))
    return chosen

CONTEXT = 10000            # total receptive field; output is len(input) - CONTEXT
_ENCODE = np.zeros((256, 4), np.float32)
for _i, _b in enumerate("ACGT"):
    _ENCODE[ord(_b), _i] = 1.0
    _ENCODE[ord(_b.lower()), _i] = 1.0


def one_hot(seq: str) -> np.ndarray:
    """Encode a sequence as (4, L) float32. Anything but ACGT/acgt maps to all-zero."""
    return _ENCODE[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)].T.copy()


class Ensemble:
    """The five SpliceAI models, averaged.

    Inputs are (B, 4, L); outputs are (B, 3, L - 10000) giving P(neither),
    P(acceptor), P(donor) for each position that had full context.
    """

    def __init__(self, models_dir, threads=None, onnx_dir=None, concurrent=True):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)           # warnings only
        t = time.perf_counter()
        paths = ensure_onnx(models_dir, onnx_dir)

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        total = int(threads) if threads else default_threads()
        # Running the five models concurrently on Python threads beats running
        # them one after another with all cores each: a 32-channel convolution
        # does not spread well over many threads, but five independent models
        # do. ONNX Runtime releases the GIL inside run(), so this is real
        # parallelism, and unlike -j it costs no extra memory.
        # Measured on 12 logical cores: 209 ms sequential vs 175 ms concurrent.
        # Run W models at a time with P threads each, keeping W*P near the
        # thread budget. Naively using ceil(total/5) threads per model and
        # always running all five oversubscribes badly when the budget is
        # small: 2 threads became 5 models x 1 thread = 5 threads, which is
        # why -j5 (2 threads per worker) was slower than -j1.
        workers = min(len(MODEL_NAMES), max(1, total))
        per = max(1, -(-total // workers))
        self.concurrent = bool(concurrent) and workers > 1
        self._workers = workers if self.concurrent else 1
        if not self.concurrent:
            per = total
        opts.intra_op_num_threads = per
        opts.inter_op_num_threads = 1
        # Per-session thread pools with spinning disabled. Idle ORT workers spin
        # by default, so five pools on one machine oversubscribe every core and
        # cost ~13x; turning the spin off fixes that without a process-wide pool.
        # A single global pool would be faster still for sequential execution,
        # but it cannot be reconfigured once created and it serialises the
        # concurrent path, which is faster than sequential anyway.
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        log.info("models: %d sessions, %d threads each, %d at a time (%d threads total)",
                 len(paths), per, self._workers, per * self._workers)
        self.sessions = [
            ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
            for p in paths
        ]
        log.info("models: ready (%.1fs)", time.perf_counter() - t)


    @property
    def _pool(self):
        """Lazily created; one worker per model, reused across calls."""
        from concurrent.futures import ThreadPoolExecutor

        pool = getattr(self, "_thread_pool", None)
        if pool is None:
            pool = ThreadPoolExecutor(max_workers=self._workers,
                                      thread_name_prefix="fsai-model")
            self._thread_pool = pool
        return pool

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (B, 4, L) -> (B, 3, L - CONTEXT), averaged over the five models."""
        if x.ndim == 2:
            x = x[None]
        x = np.ascontiguousarray(x, np.float32)
        if self.concurrent:
            outs = list(self._pool.map(lambda s: s.run(None, {"seq": x})[0],
                                       self.sessions))
        else:
            outs = [s.run(None, {"seq": x})[0] for s in self.sessions]
        acc = outs[0]
        for y in outs[1:]:
            acc = acc + y
        return acc / len(outs)

    def tile(self, x: np.ndarray, tile: int = 100_000) -> np.ndarray:
        """Score a long sequence in overlapping tiles.

        The 10kb context is paid once per tile instead of once per position, so
        the cost per scored base falls by ~2 orders of magnitude relative to
        calling predict() on a separate window for every position.
        """
        if x.ndim == 3:
            x = x[0]
        n_out = x.shape[1] - CONTEXT
        if n_out <= 0:
            raise ValueError("sequence shorter than the 10kb context")
        if n_out <= tile:
            return self.predict(x)[0]
        out = np.empty((3, n_out), np.float32)
        for start in range(0, n_out, tile):
            stop = min(start + tile, n_out)
            chunk = x[:, start : stop + CONTEXT]
            out[:, start:stop] = self.predict(chunk)[0]
        return out
