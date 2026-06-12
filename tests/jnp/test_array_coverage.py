"""Docstring."""

from ml_switcheroo.jnp.array import ndarray, _wrap
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.config import config
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device
import numpy as np


def test_array_iter_empty() -> None:
    """Docstring."""
    config.eager_mode = True
    t = Tensor(data=np.array([]), shape=(0,), dtype=DType.Float32, device=Device("cpu"))
    arr = ndarray(t)
    res = list(arr)
    assert len(res) == 0


def test_wrap_tuple() -> None:
    """Docstring."""
    t = Tensor(
        data=np.array([1]), shape=(1,), dtype=DType.Float32, device=Device("cpu")
    )
    w = _wrap((t, t))
    assert isinstance(w, tuple)
    assert isinstance(w[0], ndarray)


def test_wrap_list() -> None:
    """Docstring."""
    t = Tensor(
        data=np.array([1]), shape=(1,), dtype=DType.Float32, device=Device("cpu")
    )
    w = _wrap([t, t])
    assert isinstance(w, list)
    assert isinstance(w[0], ndarray)
