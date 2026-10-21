"""Tests for test_tracing_state."""

from __future__ import annotations

# ruff: noqa: E501
import sys
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.tracing.state import TracingState
from tests.testing_utils import as_callable, as_node, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_tracing_state() -> None:
    """Test function test_tracing_state."""
    state = TracingState()
    assert not as_tensor(state.is_tracing)
    graph = state.start_tracing("MyGraph")
    assert as_tensor(state.is_tracing)
    assert state.active_graph is graph

    class MockNode:
        """Test class MockNode."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.id = "1"
            self.source_ast_ref: EagerValue = None
            self.domain = ""
            self.stream: EagerValue = None

    node = MockNode()
    state.add_node(as_node(node))
    assert node.source_ast_ref is not None
    assert node.domain == "MyGraph"
    if "ml_switcheroo_compiler.core.config" in sys.modules:
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        old_stream = getattr(config, "current_stream", "default")
        config.current_stream = "my_stream"
        node2 = MockNode()
        node2.id = "2"
        state.add_node(as_node(node2))
        assert node2.stream == "my_stream"
        config.current_stream = old_stream
    stopped_graph = state.stop_tracing()
    assert stopped_graph is graph
    assert not state.is_tracing
    state.add_node(as_node(MockNode()))


def test_enrich_ast_and_domain_extra() -> None:
    """Test function test_enrich_ast_and_domain_extra."""
    state = TracingState()
    state.start_tracing()

    class MockNode:
        """Test class MockNode."""

        def __init__(self) -> None:
            """Test function __init__."""
            self.source_ast_ref = "existing"
            self.domain = "existing_domain"

    node = MockNode()
    as_callable(state._enrich_ast_and_domain)(node)
    assert node.source_ast_ref == "existing"
    assert node.domain == "existing_domain"
    state.stop_tracing()


def test_enrich_stream_no_config(mocker: EagerTensorProtocol) -> None:
    """Test function test_enrich_stream_no_config.

    Args:
        mocker: Test parameter `mocker`.
    """
    old_config = sys.modules.pop("ml_switcheroo_compiler.core.config", None)
    state = TracingState()

    class MockNode:
        """Test class MockNode."""

        stream = "existing"

    node = MockNode()
    as_callable(state._enrich_stream)(node)
    assert node.stream == "existing"

    if old_config is not None:
        sys.modules["ml_switcheroo_compiler.core.config"] = old_config
