"""Tests for test_wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_frontend_utils_missing() -> None:
    """Test function test_frontend_utils_missing."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        """Test class MockTensor."""

        shape = (10,)

    assert cast("Callable[..., EagerTensorProtocol]", _multinomial_shape)(cast("Callable[..., EagerTensorProtocol]", MockTensor)(), size=5) == (5,)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)
