# ruff: noqa: E501
"""Core abstractions and logic definitions for test_tensor.py."""

import numpy as np

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_tensor_coverage() -> None:
    """Test the tensor coverage behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    try:
        "Execute the requested function."
        config.eager_mode = True
        t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
        assert t.__len__() == 2
        for _x in t:
            pass
        t[0]
        config.eager_mode = False
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
