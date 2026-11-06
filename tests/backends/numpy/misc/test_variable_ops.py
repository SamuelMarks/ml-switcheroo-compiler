"""Tests for numpy eager variable ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numpy.eager.variable_ops import (
    _np_assign,
    _np_assign_variable,
    _np_bitcast,
    _np_cast,
    _np_read_variable,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_assign() -> None:
    """Test function test_assign."""
    res = _np_assign(ensure_eager_tensor(np), 1, 2)
    assert res == 2


def test_cast() -> None:
    """Test function test_cast."""
    x = np.array([1.5, 2.5])
    res = _np_cast(ensure_eager_tensor(np), x, "int32")
    assert ensure_eager_tensor(ensure_eager_tensor(res).dtype) == np.int32
    ensure_callable(np.testing.assert_allclose)(res, [1, 2])

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.value = val

    res2 = _np_cast(ensure_eager_tensor(np), x, ensure_eager_tensor(MockDtype("bfloat16")))
    assert ensure_eager_tensor(ensure_eager_tensor(res2).dtype) == np.float32

    res3 = _np_cast(ensure_eager_tensor(np), x, ensure_eager_tensor(MockDtype("int4")))
    assert ensure_eager_tensor(ensure_eager_tensor(res3).dtype) == np.int8


def test_bitcast() -> None:
    """Test function test_bitcast."""
    x = np.array([1.0], dtype=np.float32)
    res = _np_bitcast(ensure_eager_tensor(np), x, "int32")
    assert ensure_eager_tensor(ensure_eager_tensor(res).dtype) == np.int32

    class MockDtype:
        """Test class MockDtype."""

        def __init__(self, val: EagerValue) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.value = val

    res2 = _np_bitcast(ensure_eager_tensor(np), x, ensure_eager_tensor(MockDtype("int32")))
    assert ensure_eager_tensor(ensure_eager_tensor(res2).dtype) == np.int32


def test_read_variable() -> None:
    """Test function test_read_variable."""
    res = _np_read_variable(ensure_eager_tensor(np), 123)
    assert res == 123


def test_assign_variable() -> None:
    """Test function test_assign_variable."""
    res = _np_assign_variable(ensure_eager_tensor(np), "ref", 456)
    assert res == 456
