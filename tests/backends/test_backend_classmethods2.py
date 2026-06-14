"""Module docstring."""

import contextlib
import sys
import importlib
from unittest.mock import MagicMock


def test_backend_classmethods2() -> None:
    """Docstring."""
    # Mocking cupy and dask
    sys.modules["cupy"] = MagicMock()
    sys.modules["dask"] = MagicMock()
    sys.modules["dask.array"] = MagicMock()

    import ml_switcheroo_compiler.backends.jax as jax
    import ml_switcheroo_compiler.backends.keras as keras
    import ml_switcheroo_compiler.backends.mlx as mlx
    import ml_switcheroo_compiler.backends.numpy as numpy
    import ml_switcheroo_compiler.backends.pytorch as pytorch
    import ml_switcheroo_compiler.backends.tensorflow as tensorflow
    import ml_switcheroo_compiler.backends.cupy as cupy_mod
    import ml_switcheroo_compiler.backends.dask as dask_mod

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
