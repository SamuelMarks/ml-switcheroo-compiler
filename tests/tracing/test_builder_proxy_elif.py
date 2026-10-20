"""Tests for test_builder_proxy_elif."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_extract_proxy_inputs_elif_branch() -> None:
    """Test function test_extract_proxy_inputs_elif_branch."""
    import ml_switcheroo_compiler.tracing.builder as bmod
    from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder

    class FakeTracingState:
        """Test class FakeTracingState."""

        is_tracing = True
        active_graph = True

        def add_node(self, node: EagerValue) -> None:
            """Test function add_node.

            Args:
                node: Test parameter `node`.
            """
            pass

    old_state = typing.cast("EagerTensorProtocol", getattr(bmod, "global_tracing_state"))
    setattr(bmod, "global_tracing_state", FakeTracingState())

    class NonTensorWithId:
        """Test class NonTensorWithId."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.id = "my_proxy_id"
            self.shape = (4, 4)

    ids, shapes, first = TracingNodeBuilder.extract_proxy_inputs((NonTensorWithId(),))
    assert ids == ["my_proxy_id"]
    assert shapes == [(4, 4)]
    setattr(bmod, "global_tracing_state", old_state)
