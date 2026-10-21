"""Tests for test_core_math_ops_control."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _associative_scan
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_associative_scan() -> None:
    """Test function test_associative_scan."""

    def fn(a: EagerValue, b: EagerValue) -> EagerValue:
        """Add two eager values.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return as_tensor(a) + as_tensor(b)

    elems = np.array([1, 2, 3, 4])

    # scan over axis 0
    res = _associative_scan(np, fn, elems)
    assert as_callable(np.array_equal)(res, as_callable(np.array)([1, 3, 6, 10]))

    # 2d scan over axis 1
    elems2d = np.array([[1, 2, 3], [4, 5, 6]])
    res2d = _associative_scan(np, fn, elems2d, axis=1)
    assert as_callable(np.array_equal)(res2d, as_callable(np.array)([[1, 3, 6], [4, 9, 15]]))
