"""Tests for test_core_math_ops_control."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager.core_math_ops import _associative_scan

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_associative_scan() -> None:
    """Test function test_associative_scan."""

    def fn(a: EagerValue, b: EagerValue) -> EagerValue:
        return cast("EagerTensorProtocol", a) + cast("EagerTensorProtocol", b)

    elems = np.array([1, 2, 3, 4])

    # scan over axis 0
    res = _associative_scan(np, fn, elems)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res, cast("Callable[..., EagerTensorProtocol]", np.array)([1, 3, 6, 10]))

    # 2d scan over axis 1
    elems2d = np.array([[1, 2, 3], [4, 5, 6]])
    res2d = _associative_scan(np, fn, elems2d, axis=1)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(res2d, cast("Callable[..., EagerTensorProtocol]", np.array)([[1, 3, 6], [4, 9, 15]]))
