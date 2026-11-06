"""Tests for test_gru_bidirectional."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.ops.nn.gru import _compute_gru_gates, _sigmoid
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_mutable

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
            return ensure_eager_tensor(self)

        def __mul__(self, other: EagerValue) -> EagerValue:
            """Test function __mul__.

            Args:
                other: Test parameter `other`.
            """
            return ensure_eager_tensor(self)

        def __sub__(self, other: EagerValue) -> EagerValue:
            """Test function __sub__.

            Args:
                other: Test parameter `other`.
            """
            return ensure_eager_tensor(self)

        def __rsub__(self, other: EagerValue) -> EagerValue:
            """Test function __rsub__.

            Args:
                other: Test parameter `other`.
            """
            return ensure_eager_tensor(self)

    from ml_switcheroo_compiler.core.config import config

    orig = config.eager_mode
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    orig_tracing = state.global_tracing_state.is_tracing
    state.global_tracing_state.is_tracing = True
    orig_add_node = state.global_tracing_state.add_node
    ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(lambda node: None)
    try:
        ensure_callable(_compute_gru_gates)(
            [ensure_callable(DummyTensor)(), ensure_callable(DummyTensor)(), ensure_callable(DummyTensor)()],
            [ensure_callable(DummyTensor)(), ensure_callable(DummyTensor)(), ensure_callable(DummyTensor)()],
            ensure_callable(DummyTensor)(),
        )
        ensure_callable(_sigmoid)(ensure_callable(DummyTensor)())
    finally:
        config.eager_mode = orig
        state.global_tracing_state.is_tracing = orig_tracing
        ensure_mutable(state.global_tracing_state).add_node = ensure_eager_tensor(orig_add_node)
    assert ensure_eager_tensor(state.global_tracing_state).add_node == orig_add_node
