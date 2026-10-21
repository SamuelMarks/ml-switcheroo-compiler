"""Tests for test_gru_bidirectional."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.nn.gru import _compute_gru_gates, _sigmoid
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_gru_extras() -> None:
    """Test function test_gru_extras."""

    class DummyTensor:
        """Test class DummyTensor."""

        id = "t1"
        shape_metadata = None

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.
            """
            return as_tensor(self)

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.
            """
            return as_tensor(self)

        def __sub__(self, other: EagerValue) -> EagerValue:
            """Test function __sub__.

            Args:
                other: Test parameter `other`.
            """
            return as_tensor(self)

        def __rsub__(self, other: EagerValue) -> EagerValue:
            """Test function __rsub__.

            Args:
                other: Test parameter `other`.
            """
            return as_tensor(self)

    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    setattr(state.global_tracing_state, "add_node", lambda node: None)
    try:
        as_callable(_compute_gru_gates)(
            [as_callable(DummyTensor)(), as_callable(DummyTensor)(), as_callable(DummyTensor)()],
            [as_callable(DummyTensor)(), as_callable(DummyTensor)(), as_callable(DummyTensor)()],
            as_callable(DummyTensor)(),
        )
        as_callable(_sigmoid)(as_callable(DummyTensor)())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        setattr(state.global_tracing_state, "add_node", orig_add_node)
    assert getattr(state.global_tracing_state, "add_node") == orig_add_node
