"""Docstring."""

import pytest
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device
from ml_switcheroo.control_flow import vmap
from ml_switcheroo.core.config import config


def test_vmap_tuple_in_axes() -> None:
    """Docstring."""
    config.eager_mode = True

    def f(x: object) -> object:
        """Docstring."""
        return x

    t = Tensor(
        data=np.array([1, 2, 3]), shape=(3,), dtype=DType.Float32, device=Device("cpu")
    )
    vmaped_f = vmap(f, in_axes=(0,))
    with pytest.raises(Exception, match=".*"):
        vmaped_f(t)
