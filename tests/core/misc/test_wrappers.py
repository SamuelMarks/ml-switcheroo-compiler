"""Tests for test_wrappers."""

from __future__ import annotations

import typing

import pytest

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_frontend_utils_missing() -> None:
    """Test function test_frontend_utils_missing."""
    from ml_switcheroo_compiler.ops.creation.frontend_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    class MockTensor:
        """Test class MockTensor."""

        shape = (10,)

    assert typing.cast("EagerCallableProtocol", _multinomial_shape)(typing.cast("EagerCallableProtocol", MockTensor)(), size=5) == (5,)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)
