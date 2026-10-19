"""Tests for test_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.random_ops import RngBitGenerator, RngUniform

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_random_ops_infer_shape_coverage() -> None:
    """Test the random ops infer shape coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    res1 = cast("EagerCallableProtocol", RngBitGenerator)().infer_shape(None, (1, 2), None)
    assert res1 == (1, 2)
    res2 = cast("EagerCallableProtocol", RngUniform)().infer_shape(None, None, (3, 4), None)
    assert res2 == (3, 4)
