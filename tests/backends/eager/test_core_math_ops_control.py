"""Tests for test_core_math_ops_control."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _associative_scan
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_numeric

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_associative_scan() -> None:
    """Test function test_associative_scan."""

    def fn(a: BackendArg, b: BackendArg) -> BackendArg:
        """Add two eager values.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `fn`.
        """
        return coerce_numeric(a) + coerce_numeric(b)

    elems = np.array([1, 2, 3, 4])

    # scan over axis 0
    res = _associative_scan(np, fn, elems)
    assert np.array_equal(ensure_ndarray(res), np.array([1, 3, 6, 10]))

    # 2d scan over axis 1
    elems2d = np.array([[1, 2, 3], [4, 5, 6]])
    res2d = _associative_scan(np, fn, elems2d, axis=1)
    assert np.array_equal(ensure_ndarray(res2d), np.array([[1, 3, 6], [4, 9, 15]]))
