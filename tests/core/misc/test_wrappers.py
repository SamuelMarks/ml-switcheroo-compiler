"""Tests for test_wrappers."""

from __future__ import annotations

import typing

import pytest

from tests.testing_utils import MockTestStub

if typing.TYPE_CHECKING:
    pass


def test_frontend_utils_missing() -> None:
    """Test function test_frontend_utils_missing."""
    from ml_switcheroo_compiler.ops.creation.api_utils import Geometric, frompyfunc

    _multinomial_shape = Geometric().infer_shape

    class MockTensor(MockTestStub):
        """Test class MockTensor."""

        shape = (10,)

    assert _multinomial_shape(MockTensor(), size=5) == (5,)
    with pytest.raises(RuntimeError):
        frompyfunc(lambda x: x, 1, 1)
