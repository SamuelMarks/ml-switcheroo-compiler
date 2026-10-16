"""Tests for test_custom_linear_solve."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import _np_customlinearsolve

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_custom_linear_solve_fallback() -> None:
    # If no solve, return args[1]
    """Test function test_custom_linear_solve_fallback."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""
        return x

    res = _np_customlinearsolve(np, f, np.array([1, 2]))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 2]))
