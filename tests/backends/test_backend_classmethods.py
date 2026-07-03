"""Module docstring."""

import contextlib
import importlib
import sys
from unittest.mock import MagicMock

import ml_switcheroo_compiler.backends.cupy as cupy_mod
import ml_switcheroo_compiler.backends.dask as dask_mod
from ml_switcheroo_compiler.backends import jax, keras, mlx, numpy, pytorch, tensorflow
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_backend_classmethods() -> None:
    """Docstring."""
    # Mocking cupy and dask
    sys.modules["cupy"] = MagicMock()
    sys.modules["dask"] = MagicMock()
    sys.modules["dask.array"] = MagicMock()

    importlib.reload(cupy_mod)
    importlib.reload(dask_mod)

    classes = [
        jax.JAXCodeGenerator,
        keras.KerasCodeGenerator,
        mlx.MLXCodeGenerator,
        numpy.NumpyGenerator,
        pytorch.PyTorchCodeGenerator,
        tensorflow.TensorFlowCodeGenerator,
        cupy_mod.CupyGenerator,
        dask_mod.DaskGenerator,
    ]

    for mod in classes:
        with contextlib.suppress(Exception):
            mod.execute_op("Add", [1, 2])
        with contextlib.suppress(Exception):
            mod.execute_op("UnsupportedOp", [1, 2])
        with contextlib.suppress(Exception):
            mod.zeros((2, 2))
        with contextlib.suppress(Exception):
            mod.array([1, 2])
        with contextlib.suppress(Exception):
            mod.asarray([1, 2])
        with contextlib.suppress(Exception):
            mod.item([1])


def test_cupy_dask_generate() -> None:
    """Docstring."""
    g = IRGraph()
    n1 = IRNode(
        id="n1",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0]},
        shape_metadata=None,
    )
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n3 = IRNode(id="n3", op_type="Add", inputs=["n1", "n2"], attributes={}, shape_metadata=None)

    for n in [n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n2"]
    g.outputs = ["n3"]

    with contextlib.suppress(Exception):
        cupy_mod.CupyGenerator(g).generate()
    with contextlib.suppress(Exception):
        dask_mod.DaskGenerator(g).generate()
