"""Unit tests for the vectorization map (vmap) control flow operator."""

import numpy as np
import pytest

from ml_switcheroo.core.config import config
from ml_switcheroo.core.device import Device
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.ops.control_flow import vmap


def test_vmap_tuple_in_axes() -> None:
    """Verifies that vmap raises an exception when in_axes is specified as a tuple.

    This test configures the execution to eager mode and asserts that applying
    vmap with a tuple for in_axes (e.g., (0,)) on a single tensor argument
    raises an exception.
    """
    config.eager_mode = True

    def f(x: object) -> object:
        """An identity function used as a dummy target for testing vmap.

        Args:
        x (object): The input object to be returned

        Returns:
        object: The same input object.
        """
        return x

    t = Tensor(
        data=np.array([1, 2, 3]),
        shape=(3,),
        dtype=DType.Float32,
        device=Device("cpu"),
    )
    vmaped_f = vmap(f, in_axes=(0,))
    with pytest.raises(Exception, match=".*"):
        vmaped_f(t)
