"""Tests for test_math_logical_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_logical as logical_module


def test_math_logical_ops() -> None:
    """Test logical comparison and assert operations directly and through eager registry.

    Returns:
        None
    """
    a = np.array([1, 2, 3])
    b = np.array([1, 3, 2])

    assert np.array_equal(logical_module._np_not_equal(np, a, b), a != b)
    assert np.array_equal(numpy_eager_registry.get("NotEqual")(np, a, b), a != b)

    assert np.array_equal(logical_module._np_greater(np, a, b), a > b)
    assert np.array_equal(numpy_eager_registry.get("Greater")(np, a, b), a > b)

    assert np.array_equal(logical_module._np_greater_equal(np, a, b), a >= b)
    assert np.array_equal(numpy_eager_registry.get("GreaterEqual")(np, a, b), a >= b)

    assert np.array_equal(logical_module._np_less(np, a, b), a < b)
    assert np.array_equal(numpy_eager_registry.get("Less")(np, a, b), a < b)

    assert np.array_equal(logical_module._np_less_equal(np, a, b), a <= b)
    assert np.array_equal(numpy_eager_registry.get("LessEqual")(np, a, b), a <= b)

    bool_a = np.array([True, True, False])
    bool_b = np.array([True, False, False])

    assert np.array_equal(logical_module._np_logical_and(np, bool_a, bool_b), np.logical_and(bool_a, bool_b))
    assert np.array_equal(numpy_eager_registry.get("LogicalAnd")(np, bool_a, bool_b), np.logical_and(bool_a, bool_b))

    assert np.array_equal(logical_module._np_logical_or(np, bool_a, bool_b), np.logical_or(bool_a, bool_b))
    assert np.array_equal(numpy_eager_registry.get("LogicalOr")(np, bool_a, bool_b), np.logical_or(bool_a, bool_b))

    assert np.array_equal(logical_module._np_logical_not(np, bool_a), np.logical_not(bool_a))
    assert np.array_equal(numpy_eager_registry.get("LogicalNot")(np, bool_a), np.logical_not(bool_a))

    assert np.array_equal(logical_module._np_logical_xor(np, bool_a, bool_b), np.logical_xor(bool_a, bool_b))
    assert np.array_equal(numpy_eager_registry.get("LogicalXor")(np, bool_a, bool_b), np.logical_xor(bool_a, bool_b))

    cond = np.array([True, False, True])
    x_val = np.array([10, 20, 30])
    y_val = np.array([1, 2, 3])
    assert np.array_equal(logical_module._np_where(np, cond, x_val, y_val), np.where(cond, x_val, y_val))
    assert np.array_equal(numpy_eager_registry.get("Where")(np, cond, x_val, y_val), np.where(cond, x_val, y_val))

    # Assert passing
    assert logical_module._np_assert(np, np.array([True, True])) == 0
    assert numpy_eager_registry.get("Assert")(np, np.array([True, True])) == 0

    # Assert failing with custom data
    with pytest.raises(AssertionError) as exc_info:
        logical_module._np_assert(np, np.array([True, False]), data=["Custom msg"])
    assert "Custom msg" in str(exc_info.value)

    # Assert failing with default data
    with pytest.raises(AssertionError) as exc_info2:
        numpy_eager_registry.get("Assert")(np, np.array([False]))
    assert "Assertion failed." in str(exc_info2.value)
