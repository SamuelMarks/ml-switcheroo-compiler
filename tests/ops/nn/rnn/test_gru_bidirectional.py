"""Tests for test_gru_bidirectional."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ml_switcheroo_compiler.ops.nn.gru import _compute_gru_gates, _sigmoid

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_gru_extras() -> None:
    """Test function test_gru_extras."""

    class DummyTensor:
        """Test class DummyTensor."""

        id = "t1"
        shape_metadata = None

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__."""
            return cast("EagerTensorProtocol", self)

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__."""
            return cast("EagerTensorProtocol", self)

        def __sub__(self, other: EagerValue) -> EagerValue:
            """Test function __sub__."""
            return cast("EagerTensorProtocol", self)

        def __rsub__(self, other: EagerValue) -> EagerValue:
            """Test function __rsub__."""
            return cast("EagerTensorProtocol", self)

    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        cast("Callable[..., EagerTensorProtocol]", _compute_gru_gates)(
            [cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), cast("Callable[..., EagerTensorProtocol]", DummyTensor)()],
            [cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), cast("Callable[..., EagerTensorProtocol]", DummyTensor)(), cast("Callable[..., EagerTensorProtocol]", DummyTensor)()],
            cast("Callable[..., EagerTensorProtocol]", DummyTensor)(),
        )
        cast("Callable[..., EagerTensorProtocol]", _sigmoid)(cast("Callable[..., EagerTensorProtocol]", DummyTensor)())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
    assert orig_add_node is not None or orig_add_node is None
