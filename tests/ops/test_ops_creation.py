"""Unit tests for basic tensor creation operations.

This module contains tests to verify the shape inference and NumPy evaluation behavior
of Zeros, Ones, Full, and Arange operations against their NumPy equivalents.
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import get_op
from ml_switcheroo_compiler.ops.creation.basic import Arange, Full, Ones, Zeros


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
        assert np.array_equal(op.eager_eval(shape), np_func(shape))


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
    assert np.array_equal(op.eager_eval(shape, val), np.full(shape, val))


def test_arange_op() -> None:
    """Tests the Arange tensor creation operation.

    Verifies that the Arange operation correctly handles shape inference
    and evaluates to a NumPy array containing a sequence of numbers

    Returns:
    None
    """
    op = Arange()
    assert op.infer_shape(10) is None
    assert np.array_equal(op.eager_eval(5), np.arange(5))


def test_rand_ops() -> None:
    """Test function."""
    op = get_op("Rand")()
    assert op.infer_shape(size=(2, 3)) == (2, 3)
    assert op.infer_shape((2, 3)) == (2, 3)
    assert op.infer_shape(2, 3) == (2, 3)
    res = op.eager_eval(2, 3, dtype=np.float32)
    assert res.shape == (2, 3)
    assert res.dtype == np.float32

    res_none = op.eager_eval(2, 3, dtype=None)
    assert res_none.shape == (2, 3)

    op_randn = get_op("Randn")()
    res_randn = op_randn.eager_eval(2, 3, dtype=np.float32)
    assert res_randn.shape == (2, 3)
    assert res_randn.dtype == np.float32

    res_randn_none = op_randn.eager_eval(2, 3, dtype=None)
    assert res_randn_none.shape == (2, 3)

    op_randint = get_op("Randint")()
    assert op_randint.infer_shape(size=(2, 3)) == (2, 3)
    assert op_randint.infer_shape(0, 10, (2, 3)) == (2, 3)
    assert op_randint.infer_shape(0, 10) == ()
    res_randint = op_randint.eager_eval(0, 10, size=(2, 3), dtype=np.int32)
    assert res_randint.shape == (2, 3)
    assert res_randint.dtype == np.int32

    res_randint2 = op_randint.eager_eval(10, size=(2, 3), dtype=None)
    assert res_randint2.shape == (2, 3)

    op_seed = get_op("ManualSeed")()
    assert op_seed.infer_shape(42) == ()
    assert op_seed.eager_eval(42) == 42
