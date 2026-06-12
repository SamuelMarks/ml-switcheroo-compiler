"""Unit tests for verifying Tensor dunder (magic) methods and operations.

This module contains test cases to ensure that the Tensor class correctly implements and
executes various Python magic methods (dunders) such as arithmetic, bitwise, comparison,
unary, indexing, and type conversion operations in eager mode.
"""

import numpy as np

from ml_switcheroo.core.config import config
from ml_switcheroo.core.device import Device, DeviceType
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.tensor import Tensor


def test_tensor_dunders() -> None:
    """Verifies the correct execution of various magic (dunder) methods on the Tensor.

    class

    This test runs in eager mode and exercises arithmetic, bitwise, comparison,
    unary, indexing, in-place modification, and type conversion operations
    on Tensor instances to ensure they execute without errors

    Returns:
    None
    """
    config.eager_mode = True
    d = Device(DeviceType.CPU, 0)
    t1 = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, d)
    t2 = Tensor(np.array([4, 5, 6]), (3,), DType.Int32, d)
    t3 = Tensor(np.array([1]), (1,), DType.Int32, d)
    _ = t1 % t2
    _ = t1 & t2
    _ = t1 | t2
    _ = t1 ^ t2
    _ = t1 << t2
    _ = t1 >> t2
    _ = abs(t1)
    _ = ~t1
    _ = t1 < t2
    _ = t1 <= t2
    _ = t1 == t2
    _ = t1 != t2
    _ = t1 > t2
    _ = t1 >= t2
    _ = bool(t3)
    _ = len(t1)
    _ = list(t1)
    _ = +t1
    _ = t1[0]
    config.enable_in_place = True
    t1[0] = 5
    config.enable_in_place = False
    _ = np.array(t1)
