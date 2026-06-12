"""Docstring."""

import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device
from ml_switcheroo.jnp.array import ndarray
import ml_switcheroo.jnp as jnp
from ml_switcheroo.core.config import config


def test_array_ndarray_passthrough() -> None:
    """Docstring."""
    t = Tensor(
        data=np.array([1]), shape=(1,), dtype=DType.Float32, device=Device("cpu")
    )
    arr = ndarray(t)
    res = jnp.array(arr)
    assert res is arr


def test_meshgrid_xy() -> None:
    """Docstring."""
    config.eager_mode = True
    t = Tensor(
        data=np.array([1, 2]), shape=(2,), dtype=DType.Float32, device=Device("cpu")
    )
    arr = ndarray(t)
    res = jnp.meshgrid(arr, arr, indexing="xy")
    assert isinstance(res, tuple)
    assert len(res) == 2
