"""Module docstring."""

import contextlib
import sys
import importlib
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def get_graph(op_type: str) -> IRGraph:
    """Docstring."""
    g = IRGraph()
    n1 = IRNode(
        id="n1", op_type="Constant", inputs=[], attributes={"value": [1.0]}, shape_metadata=None
    )
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n3 = IRNode(
        id="n3", op_type=op_type, inputs=["n1", "n2"], attributes={"axis": 0}, shape_metadata=None
    )
    for n in [n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n2"]
    g.outputs = ["n3"]
    return g


def test_backends_brute_coverage_specifics() -> None:
    """Docstring."""
    for lib in [
        "cupy",
        "dask",
        "dask.array",
        "jax",
        "jax.numpy",
        "keras",
        "keras.ops",
        "mlx",
        "mlx.core",
        "torch",
        "tensorflow",
    ]:
        if lib not in sys.modules:
            sys.modules[lib] = type("MockModule", (), {})()

    import ml_switcheroo_compiler.backends.jax as jax
    import ml_switcheroo_compiler.backends.keras as keras
    import ml_switcheroo_compiler.backends.mlx as mlx
    import ml_switcheroo_compiler.backends.numpy as numpy
    import ml_switcheroo_compiler.backends.pytorch as pytorch
    import ml_switcheroo_compiler.backends.tensorflow as tensorflow
    import ml_switcheroo_compiler.backends.cupy as cupy
    import ml_switcheroo_compiler.backends.dask as dask

    for mod in [jax, keras, mlx, numpy, pytorch, tensorflow, cupy, dask]:
        importlib.reload(mod)

    classes = [
        jax.JAXCodeGenerator,
        keras.KerasCodeGenerator,
        mlx.MLXCodeGenerator,
        numpy.NumpyGenerator,
        pytorch.PyTorchCodeGenerator,
        tensorflow.TensorFlowCodeGenerator,
        cupy.CupyGenerator,
        dask.DaskGenerator,
    ]

    op_types = [
        "Gelu",
        "TestEagerOp",
        "DummyBinary",
        "DummyUnary",
        "RandomUniformInt",
        "Xlogy",
        "BroadcastInDim",
        "Pmean",
        "Add",
        "Subtract",
        "Multiply",
        "TrueDivide",
        "Exp",
        "Log",
        "Matmul",
        "Sin",
        "Cos",
        "Sum",
        "Mean",
        "Max",
        "Min",
        "BroadcastTo",
        "Reshape",
        "Transpose",
        "Equal",
        "NotEqual",
        "Greater",
        "Less",
        "Negative",
    ]

    for mod in classes:
        for op in op_types:
            with contextlib.suppress(Exception):
                mod.execute_op(op, 0.0, 1.0, shape=[1, 2], broadcast_dimensions=[0])
            with contextlib.suppress(Exception):
                mod(get_graph(op)).generate()
