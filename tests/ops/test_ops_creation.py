"""Unit tests for basic tensor creation operations.

This module contains tests to verify the shape inference and NumPy evaluation behavior
of Zeros, Ones, Full, and Arange operations against their NumPy equivalents.
"""

import numpy as np

from ml_switcheroo.ops.creation.basic import (
    Arange,
    Full,
    Ones,
    Zeros,
)


def test_creation_ops() -> None:
    """Tests the Zeros and Ones tensor creation operations.

    Verifies that both Zeros and Ones operations correctly infer the target
    shape and evaluate to the expected NumPy arrays

    Returns:
    None
    """
    shape = (2, 3)

    ops = [
        (Zeros(), np.zeros),
        (Ones(), np.ones),
    ]

    for op, np_func in ops:
        assert op.infer_shape(shape) == shape
        assert np.array_equal(op.numpy_eval(shape), np_func(shape))




def test_full_op() -> None:
    """Tests the Full tensor creation operation.

    Verifies that the Full operation correctly infers the target shape
    and evaluates to a NumPy array filled with the specified value

    Returns:
    None
    """
    op = Full()
    shape = (2, 2)
    val = 5.0
    assert op.infer_shape(shape, val) == shape
    assert np.array_equal(op.numpy_eval(shape, val), np.full(shape, val))


def test_arange_op() -> None:
    """Tests the Arange tensor creation operation.

    Verifies that the Arange operation correctly handles shape inference
    and evaluates to a NumPy array containing a sequence of numbers

    Returns:
    None
    """
    op = Arange()
    assert op.infer_shape(10) is None
    assert np.array_equal(op.numpy_eval(5), np.arange(5))
