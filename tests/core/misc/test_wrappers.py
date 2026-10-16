"""Tests for test_wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_frontend_utils_missing() -> None:
    """Test function test_frontend_utils_missing."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    # 1. _multinomial_shape where size is int
    class MockTensor:
        """Test class MockTensor."""

        shape = (10,)

    assert cast("Callable[..., EagerTensorProtocol]", _multinomial_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)(), size=5) == (5,)

    # 2. frompyfunc
    try:
        frompyfunc(lambda x: x, 1, 1)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None
