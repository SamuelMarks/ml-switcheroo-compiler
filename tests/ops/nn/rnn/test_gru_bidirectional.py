"""Tests for test_gru_bidirectional."""

from __future__ import annotations

import typing

import pytest

from ml_switcheroo_compiler.ops.nn.gru import _compute_gru_gates, _sigmoid
from tests.testing_utils import MockTensorStub, Tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_gru_extras() -> None:
    """Test function test_gru_extras."""

    class DummyTensor(MockTensorStub):
        """Test class DummyTensor."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.id = "t1"
            self.shape_metadata = None

        def __add__(self, other: BackendArg) -> Tensor:
            """Test function __add__.

            Args:
                other: Test parameter `other`.

            Returns:
                Tensor: Computed output value from `__add__`.
            """
            return self

        def __mul__(self, other: BackendArg) -> Tensor:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.

            Returns:
                Tensor: Computed output value from `__mul__`.
            """
            return self

        def __sub__(self, other: BackendArg) -> Tensor:
            """Test function __sub__.

            Args:
                other: Test parameter `other`.

            Returns:
                Tensor: Computed output value from `__sub__`.
            """
            return self

        def __rsub__(self, other: BackendArg) -> Tensor:
            """Test function __rsub__.

            Args:
                other: Test parameter `other`.

            Returns:
                Tensor: Computed output value from `__rsub__`.
            """
            return self

    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", lambda node: None, raising=False)
    try:
        _compute_gru_gates(
            [DummyTensor(), DummyTensor(), DummyTensor()],
            [DummyTensor(), DummyTensor(), DummyTensor()],
            DummyTensor(),
        )
        _sigmoid(DummyTensor())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        pytest.MonkeyPatch().setattr(state.global_tracing_state, "add_node", orig_add_node, raising=False)
    assert state.global_tracing_state.add_node == orig_add_node
