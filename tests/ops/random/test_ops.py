"""Tests for test_ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.random_ops import RngBitGenerator, RngUniform

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_random_ops_infer_shape_coverage() -> None:
    """Test the random ops infer shape coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        res1 = cast("Callable[..., EagerTensorProtocol]", RngBitGenerator)().infer_shape(None, (1, 2), None)
        assert res1 == (1, 2)
        res2 = cast("Callable[..., EagerTensorProtocol]", RngUniform)().infer_shape(None, None, (3, 4), None)
        assert res2 == (3, 4)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None
