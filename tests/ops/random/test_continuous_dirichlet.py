"""Tests for test_continuous_dirichlet."""

from __future__ import annotations

import typing

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.random.continuous.dirichlet import dirichlet
from tests.testing_utils import MockTensorStub

if typing.TYPE_CHECKING:
    pass


def test_dirichlet(mocker: MockerFixture) -> None:
    # Mock global_tracing_state.add_node inside random.state
    """Test function test_dirichlet.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.add_node")

    class MockTensor(MockTensorStub):
        """Test class MockTensor."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = type("M", (), {"id": "1"})()
            self.shape = ()

    t = MockTensor()
    res = dirichlet(t, t)
    assert tuple(res.shape) == ()

    res2 = dirichlet(t, t, shape=(2, 3), dtype=DType.Float64)
    assert tuple(res2.shape) == (2, 3)
