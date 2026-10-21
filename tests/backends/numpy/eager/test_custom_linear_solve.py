"""Tests for test_custom_linear_solve."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_customlinearsolve
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_custom_linear_solve_fallback() -> None:
    # If no solve, return args[1]
    """Test function test_custom_linear_solve_fallback."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return x

    res = _np_customlinearsolve(np, f, np.array([1, 2]))
    assert as_callable(np.array_equal)(res, as_callable(np.array)([1, 2]))
