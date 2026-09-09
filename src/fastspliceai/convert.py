"""Convert the Keras/HDF5 SpliceAI models to ONNX without importing TensorFlow.

The weights are read straight out of the HDF5 files with h5py, so installing
fast-splice-ai does not drag in TensorFlow or Keras.

Two things happen during conversion:

1.  Batch-norm is folded into a per-channel scale/offset.  Every BN in SpliceAI
    is followed by a ReLU, so the folded form is exact.

2.  Every convolution is rewritten from ``padding='same'`` to ``padding='valid'``.
    The network is fully convolutional with a total receptive field of +/-5000,
    so a 'same'-padded model on a length-L window computes L positions and then
    throws away all but the middle L-10000 of them.  In 'valid' form each layer
    only computes the positions that can still influence an output, and the
    trailing Cropping1D becomes a no-op.  The result is bit-identical and costs
    about 1.4x less arithmetic.  It also makes the graph length-agnostic, which
    is what lets `engine.tile()` amortise the 10kb context over a long window.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import logging
import time

import h5py
import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H
from onnx import numpy_helper as NH

log = logging.getLogger("fastspliceai.convert")

BN_EPS = 1e-3
# bump when the emitted graph changes, so stale cached .onnx files are rebuilt
CONVERTER_VERSION = 2
MODEL_NAMES = tuple("spliceai{}.h5".format(i) for i in range(1, 6))


def _layers(h5):
    cfg = h5.attrs["model_config"]
    if isinstance(cfg, bytes):
        cfg = cfg.decode()
    return json.loads(cfg)["config"]


def _inbound(layer):
    nodes = layer.get("inbound_nodes") or []
    return [b[0] for b in nodes[0]] if nodes else []


def _half_span(layer):
    """Half of the width a 'same' conv would pad, i.e. how much a valid conv shrinks."""
    k = layer["config"]["kernel_size"][0]
    d = layer["config"]["dilation_rate"][0]
    return d * (k - 1) // 2 if layer["config"]["padding"] == "same" else 0


def plan(h5_path):
    """Return (ops, shrink, total_shrink).

    ``shrink[name]`` is how many positions have been trimmed from *each* end of
    that tensor relative to the model input, in the valid-convolution rewrite.
    """
    with h5py.File(h5_path, "r") as f:
        cfg = _layers(f)
    shrink, ops = {}, []
    for l in cfg["layers"]:
        cls, name, inb = l["class_name"], l["config"]["name"], _inbound(l)
        if cls == "InputLayer":
            shrink[name] = 0
        elif cls == "Conv1D":
            shrink[name] = shrink[inb[0]] + _half_span(l)
        elif cls == "Cropping1D":
            # already handled by the valid rewrite; keep the tensor as-is
            shrink[name] = shrink[inb[0]]
        elif cls == "Add":
            shrink[name] = max(shrink[i] for i in inb)
        else:
            shrink[name] = shrink[inb[0]]
        ops.append((cls, name, inb, l))
    return ops, shrink, shrink[cfg["output_layers"][0][0]]


def convert(h5_path, onnx_path):
    """Write a length-agnostic valid-convolution ONNX graph. Returns onnx_path."""
    ops, shrink, total = plan(h5_path)
    with h5py.File(h5_path, "r") as f:
        cfg, W = _layers(f), f["model_weights"]
        nodes, inits = [], []

        def const(name, arr):
            inits.append(NH.from_array(np.ascontiguousarray(arr), name))
            return name

        def trim(src, amount, tag):
            """Symmetrically drop `amount` positions from each end of a tensor."""
            if amount == 0:
                return src
            out = "{}_trim{}".format(tag, amount)
            nodes.append(
                H.make_node(
                    "Slice",
                    [
                        src,
                        const(out + "_s", np.array([amount], np.int64)),
                        const(out + "_e", np.array([-amount], np.int64)),
                        const(out + "_a", np.array([2], np.int64)),
                    ],
                    [out],
                )
            )
            return out

        for cls, name, inb, l in ops:
            c = l["config"]
            if cls == "InputLayer":
                continue
            if cls == "Conv1D":
                g = W[name]
                # keras (k, Cin, Cout) -> onnx (Cout, Cin, k)
                k = np.asarray(g[name + "/kernel:0"], np.float32).transpose(2, 1, 0)
                b = np.asarray(g[name + "/bias:0"], np.float32)
                out = name if c["activation"] == "linear" else name + "_pre"
                nodes.append(
                    H.make_node(
                        "Conv",
                        [inb[0], const(name + "_w", k), const(name + "_b", b)],
                        [out],
                        name=name,
                        dilations=[c["dilation_rate"][0]],
                        kernel_shape=[k.shape[2]],
                        pads=[0, 0],          # <- the valid rewrite
                        strides=[1],
                    )
                )
                if c["activation"] == "softmax":
                    nodes.append(H.make_node("Softmax", [out], [name], axis=1))
                elif c["activation"] != "linear":
                    raise ValueError("unsupported activation " + c["activation"])
            elif cls == "BatchNormalization":
                g = W[name]
                ga, be, mu, va = (
                    np.asarray(g["{}/{}:0".format(name, x)], np.float32)
                    for x in ("gamma", "beta", "moving_mean", "moving_variance")
                )
                sc = (ga / np.sqrt(va + BN_EPS)).astype(np.float32)
                of = (be - mu * sc).astype(np.float32)
                nodes.append(H.make_node("Mul", [inb[0], const(name + "_s", sc[None, :, None])], [name + "_m"]))
                nodes.append(H.make_node("Add", [name + "_m", const(name + "_o", of[None, :, None])], [name]))
            elif cls == "Activation":
                if c["activation"] != "relu":
                    raise ValueError("unsupported activation " + c["activation"])
                nodes.append(H.make_node("Relu", [inb[0]], [name], name=name))
            elif cls == "Add":
                # the two branches have shrunk by different amounts; re-align them
                srcs = [trim(i, shrink[name] - shrink[i], name + "_" + i) for i in inb]
                nodes.append(H.make_node("Add", srcs, [name], name=name))
            elif cls == "Cropping1D":
                nodes.append(H.make_node("Identity", [inb[0]], [name], name=name))
            else:
                raise ValueError("unsupported layer " + cls)

        out_name = cfg["output_layers"][0][0]
        # the keras input tensor is called 'input_1'; expose it as 'seq'
        for n in nodes:
            n.input[:] = ["seq" if i == "input_1" else i for i in n.input]
        graph = H.make_graph(
            nodes,
            "fastspliceai",
            [H.make_tensor_value_info("seq", TP.FLOAT, ["B", 4, "L"])],
            [H.make_tensor_value_info(out_name, TP.FLOAT, ["B", 3, "Lout"])],
            inits,
        )
        m = H.make_model(graph, opset_imports=[H.make_opsetid("", 13)])
        m.ir_version = 9
        m.doc_string = "fast-splice-ai valid-conv rewrite; context={}".format(2 * total)
        onnx.checker.check_model(m)
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(m, onnx_path)
    return onnx_path


def cache_dir():
    base = os.environ.get("FASTSPLICEAI_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "fastspliceai"
    )
    return Path(base)


def ensure_onnx(models_dir, dest=None):
    """Convert the five .h5 models on first use; return the list of .onnx paths."""
    models_dir, dest = Path(models_dir), Path(dest or cache_dir())
    out = []
    for n in MODEL_NAMES:
        src = models_dir / n
        dst = dest / "{}.v{}.onnx".format(n[:-3], CONVERTER_VERSION)
        if not src.exists():
            raise FileNotFoundError("missing SpliceAI model: {}".format(src))
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            log.info("converting %s to ONNX (first run only)", src.name)
            t = time.perf_counter()
            convert(src, dst)
            log.info("  %s written in %.1fs", dst.name, time.perf_counter() - t)
        out.append(dst)
    return out
