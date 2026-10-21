"""Tests for test_continuous_dirichlet."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.random.continuous.dirichlet import dirichlet
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_dirichlet(mocker: EagerTensorProtocol) -> None:
    # Mock global_tracing_state.add_node inside random.state
    """Test function test_dirichlet.

    Args:
        mocker: Test parameter `mocker`.
    """
    mocker.patch("ml_switcheroo_compiler.random.state.global_tracing_state.add_node")

    class MockTensor:
        """Test class MockTensor."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.data = type("M", (), {"id": "1"})()
            self.shape = ()

    t = MockTensor()
    res = as_callable(dirichlet)(t, t)
    assert res is not None

    res2 = as_callable(dirichlet)(t, t, shape=(2, 3), dtype=DType.Float64)
    assert res2 is not None
