"""Tests for numpy eager math logical ops."""

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.math_logical import (
    _np_assert,
    _np_greater,
    _np_greater_equal,
    _np_less,
    _np_less_equal,
    _np_logical_and,
    _np_logical_not,
    _np_logical_or,
    _np_logical_xor,
    _np_not_equal,
    _np_where,
)


def test_not_equal() -> None:
    """Test not equal.

    Returns:
        None
    """
    assert _np_not_equal(np, 1, 2)


def test_greater() -> None:
    """Test greater.

    Returns:
        None
    """
    assert _np_greater(np, 2, 1)


def test_greater_equal() -> None:
    """Test greater equal.

    Returns:
        None
    """
    assert _np_greater_equal(np, 2, 2)


def test_less() -> None:
    """Test less.

    Returns:
        None
    """
    assert _np_less(np, 1, 2)


def test_less_equal() -> None:
    """Test less equal.

    Returns:
        None
    """
    assert _np_less_equal(np, 2, 2)


def test_logical_and() -> None:
    """Test logical and.

    Returns:
        None
    """
    assert _np_logical_and(np, True, True)


def test_logical_or() -> None:
    """Test logical or.

    Returns:
        None
    """
    assert _np_logical_or(np, True, False)


def test_logical_not() -> None:
    """Test logical not.

    Returns:
        None
    """
    assert _np_logical_not(np, False)


def test_logical_xor() -> None:
    """Test logical xor.

    Returns:
        None
    """
    assert _np_logical_xor(np, True, False)


def test_where() -> None:
    """Test where.

    Returns:
        None
    """
    res = _np_where(np, np.array([True, False]), 1, 0)
    assert np.array_equal(res, [1, 0])


def test_assert() -> None:
    """Test assert.

    Returns:
        None
    """
    _np_assert(np, True)
    with pytest.raises(AssertionError, match="Assertion failed."):
        _np_assert(np, False)
