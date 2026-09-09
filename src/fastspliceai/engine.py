"""ONNX Runtime inference for the five SpliceAI models."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from .convert import ensure_onnx

CONTEXT = 10000            # total receptive field; output is len(input) - CONTEXT
_ENCODE = np.zeros((256, 4), np.float32)
for _i, _b in enumerate("ACGT"):
    _ENCODE[ord(_b), _i] = 1.0
    _ENCODE[ord(_b.lower()), _i] = 1.0


def one_hot(seq: str) -> np.ndarray:
    """Encode a sequence as (4, L) float32. Anything but ACGT/acgt maps to all-zero."""
    return _ENCODE[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)].T.copy()


_GLOBAL_POOL = {"size": None, "failed": False}


def _use_global_pool(ort, n_threads):
    """Ask ORT for one process-wide thread pool, once. Returns True if in use.

    Without this each session owns a pool whose idle workers spin, so five
    models means five pools fighting over the same cores (~13x slower).

    The pool is created once per process and its size is then fixed: ORT rejects
    per-session threads for the rest of the process's life. A later Ensemble
    asking for a different thread count therefore still uses the global pool,
    and we say so rather than crashing.
    """
    if _GLOBAL_POOL["failed"]:
        return False
    size = _GLOBAL_POOL["size"]
    if size is not None:
        if size != n_threads:
            warnings.warn(
                "ONNX Runtime's thread pool is process-wide and was already created "
                "with {} threads; this Ensemble's threads={} is ignored. Set the "
                "thread count on the first Ensemble, or use separate processes."
                .format(size, n_threads),
                RuntimeWarning, stacklevel=3)
        return True
    try:
        ort.set_global_thread_pool_sizes(n_threads, 1)
    except Exception:                                # pragma: no cover
        _GLOBAL_POOL["failed"] = True
        return False
    _GLOBAL_POOL["size"] = n_threads
    return True


class Ensemble:
    """The five SpliceAI models, averaged.

    Inputs are (B, 4, L); outputs are (B, 3, L - 10000) giving P(neither),
    P(acceptor), P(donor) for each position that had full context.
    """

    def __init__(self, models_dir, threads=None, precision="fp32", onnx_dir=None):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)           # warnings only
        if precision not in ("fp32", "int8"):
            raise ValueError("precision must be 'fp32' or 'int8'")
        self.precision = precision
        paths = ensure_onnx(models_dir, onnx_dir)
        if precision == "int8":
            paths = [self._quantize(p) for p in paths]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(threads or os.cpu_count() or 1)
        opts.inter_op_num_threads = 1
        # The five models run one after another, but each session would otherwise
        # own a thread pool whose workers *spin* while waiting for work. Five
        # spinning pools on one machine oversubscribe every core and cost ~13x.
        # Sharing one pool across the sessions is the fix; disabling the spin is
        # the fallback for runtimes that do not support a global pool.
        n_threads = opts.intra_op_num_threads
        self.sessions = None
        if _use_global_pool(ort, n_threads):
            opts.use_per_session_threads = False
            try:
                self.sessions = [
                    ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                    for p in paths
                ]
            except Exception:                        # pragma: no cover
                _GLOBAL_POOL["failed"] = True
                opts.use_per_session_threads = True
        if self.sessions is None:
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            self.sessions = [
                ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
                for p in paths
            ]

    @staticmethod
    def _quantize(fp32_path):
        """Static INT8 quantisation. NOTE: not bit-identical - see README."""
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )

        fp32_path = Path(fp32_path)
        out = fp32_path.with_name(fp32_path.stem + ".int8.onnx")
        if out.exists() and out.stat().st_mtime >= fp32_path.stat().st_mtime:
            return out

        rng = np.random.default_rng(0)

        class _Rand(CalibrationDataReader):
            """Calibrate on random sequence. Splice signals are sparse, so random
            DNA exercises the same activation ranges as real DNA for this net."""

            def __init__(self):
                batches = []
                for _ in range(8):
                    x = np.zeros((1, 4, CONTEXT + 1001), np.float32)
                    x[0, rng.integers(0, 4, x.shape[2]), np.arange(x.shape[2])] = 1
                    batches.append({"seq": x})
                self.it = iter(batches)

            def get_next(self):
                return next(self.it, None)

        quantize_static(
            str(fp32_path),
            str(out),
            _Rand(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (B, 4, L) -> (B, 3, L - CONTEXT), averaged over the five models."""
        if x.ndim == 2:
            x = x[None]
        x = np.ascontiguousarray(x, np.float32)
        acc = None
        for s in self.sessions:
            y = s.run(None, {"seq": x})[0]
            acc = y if acc is None else acc + y
        return acc / len(self.sessions)

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
