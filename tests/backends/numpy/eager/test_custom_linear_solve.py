"""Tests for test_custom_linear_solve."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import _np_customlinearsolve

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_custom_linear_solve_fallback() -> None:
    # If no solve, return args[1]
    """Test function test_custom_linear_solve_fallback."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return x

    res = _np_customlinearsolve(np, f, np.array([1, 2]))
    assert np.array_equal(ensure_ndarray(res), np.array([1, 2]))
