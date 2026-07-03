"""Test approx max/min k."""

import numpy as np

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import config


def test_approx_k() -> object:
    """Function docstring."""
    config.eager_mode = True
    x = ops.array(np.array([1, 4, 3, 2, 5]))

    val, idx = ops.approx_max_k(x, 2)
    np.testing.assert_array_equal(val.data, np.array([5, 4]))
    np.testing.assert_array_equal(idx.data, np.array([4, 1]))

    val, idx = ops.approx_min_k(x, 2)
    np.testing.assert_array_equal(val.data, np.array([1, 2]))
    np.testing.assert_array_equal(idx.data, np.array([0, 3]))
