"""Tests for numpy eager variable ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import (
    _np_assign,
    _np_assign_variable,
    _np_bitcast,
    _np_cast,
    _np_read_variable,
)

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_assign() -> None:
    """Test function test_assign."""
    res = _np_assign(np, 1, 2)
    assert res == 2


def test_cast() -> None:
    """Test function test_cast."""
    x = np.array([1.5, 2.5])
    res = _np_cast(np, x, "int32")
    assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int32
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res, [1, 2])

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.value = val

    res2 = _np_cast(np, x, MockDtype("bfloat16"))
    assert cast("EagerTensorProtocol", getattr(res2, "dtype")) == np.float32

    res3 = _np_cast(np, x, MockDtype("int4"))
    assert cast("EagerTensorProtocol", getattr(res3, "dtype")) == np.int8


def test_bitcast() -> None:
    """Test function test_bitcast."""
    x = np.array([1.0], dtype=np.float32)
    res = _np_bitcast(np, x, "int32")
    assert cast("EagerTensorProtocol", getattr(res, "dtype")) == np.int32

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.value = val

    res2 = _np_bitcast(np, x, MockDtype("int32"))
    assert cast("EagerTensorProtocol", getattr(res2, "dtype")) == np.int32


def test_read_variable() -> None:
    """Test function test_read_variable."""
    res = _np_read_variable(np, 123)
    assert res == 123


def test_assign_variable() -> None:
    """Test function test_assign_variable."""
    res = _np_assign_variable(np, "ref", 456)
    assert res == 456
