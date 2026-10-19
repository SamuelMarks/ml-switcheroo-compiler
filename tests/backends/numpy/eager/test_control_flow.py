"""Tests for test_control_flow."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.control_flow import _np_associative_scan

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_np_associative_scan() -> None:
    """Test function test_np_associative_scan."""

    def fn(a: EagerValue, b: EagerValue) -> EagerValue:
        """Add two eager values.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        return cast("EagerTensorProtocol", a) + cast("EagerTensorProtocol", b)

    elems = np.array([1, 2, 3, 4])

    # scan over axis 0
    res = _np_associative_scan(np, fn, elems)
    assert cast("EagerCallableProtocol", np.array_equal)(res, cast("EagerCallableProtocol", np.array)([1, 3, 6, 10]))

    # 2d scan over axis 1
    elems2d = np.array([[1, 2, 3], [4, 5, 6]])
    res2d = _np_associative_scan(np, fn, elems2d, axis=1)
    assert cast("EagerCallableProtocol", np.array_equal)(res2d, cast("EagerCallableProtocol", np.array)([[1, 3, 6], [4, 9, 15]]))
