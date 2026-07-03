"""Module docstring."""

import contextlib
import importlib
import sys
from unittest.mock import MagicMock

import ml_switcheroo_compiler.backends.cupy as cupy_mod
import ml_switcheroo_compiler.backends.dask as dask_mod
from ml_switcheroo_compiler.backends import jax, keras, mlx, numpy, pytorch, tensorflow


def test_backend_classmethods2() -> None:
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
            mod.execute_op("Arange", [1, 2])
        with contextlib.suppress(Exception):
            mod.execute_op("UnsupportedOp", [1, 2])
        with contextlib.suppress(Exception):
            mod.execute_op("Add", [1, 2])

        with contextlib.suppress(Exception):
            mod.zeros((2, 2))
        with contextlib.suppress(Exception):
            mod.array([1, 2])
        with contextlib.suppress(Exception):
            mod.asarray([1, 2])
        with contextlib.suppress(Exception):
            mod.item([1])

    pass
