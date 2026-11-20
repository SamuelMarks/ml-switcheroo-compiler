"""Tests for test_tracing_state."""

from __future__ import annotations

import sys

from pytest_mock import MockerFixture

from ml_switcheroo_compiler.tracing.state import TracingState
from tests.testing_utils import MockNodeStub


def test_tracing_state() -> None:
    """Test function test_tracing_state."""
    state = TracingState()
    initial_tracing = state.is_tracing
    assert not initial_tracing
    graph = state.start_tracing("MyGraph")
    started_tracing = state.is_tracing
    assert started_tracing
    assert state.active_graph is graph

    class MockNode(MockNodeStub):
        """Test class MockNode."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__()
            self.id = "1"
            self.source_ast_ref = None
            self.domain = ""
            self.stream = None

    node = MockNode()
    state.add_node(node)
    assert node.source_ast_ref is not None
    assert node.domain == "MyGraph"
    if "ml_switcheroo_compiler.core.config" in sys.modules:
        from ml_switcheroo_compiler.core.config import config as active_config

        old_stream = active_config.current_stream
        active_config.current_stream = "my_stream"
        node2 = MockNode()
        node2.id = "2"
        state.add_node(node2)
        assert node2.stream == "my_stream"
        active_config.current_stream = old_stream
    stopped_graph = state.stop_tracing()
    assert stopped_graph is graph
    stopped_tracing = state.is_tracing
    assert not stopped_tracing
    state.add_node(MockNode())


def test_enrich_ast_and_domain_extra() -> None:
    """Test function test_enrich_ast_and_domain_extra."""
    state = TracingState()
    state.start_tracing()

    class MockNode(MockNodeStub):
        """Test class MockNode."""

        def __init__(self) -> None:
            """Test function __init__."""
            super().__init__()
            self.source_ast_ref = "existing"
            self.domain = "existing_domain"

    node = MockNode()
    state._enrich_ast_and_domain(node)
    assert node.source_ast_ref == "existing"
    assert node.domain == "existing_domain"
    state.stop_tracing()


def test_enrich_stream_no_config(mocker: MockerFixture) -> None:
    """Test function test_enrich_stream_no_config.

    Args:
        mocker (MockerFixture): Test parameter `mocker`.
    """
    old_config = sys.modules.pop("ml_switcheroo_compiler.core.config", None)
    state = TracingState()

    class MockNode(MockNodeStub):
        """Test class MockNode."""

        def __init__(self) -> None:
            """Initialize stub attributes."""
            super().__init__()
            self.stream = "existing"

    node = MockNode()
    state._enrich_stream(node)
    assert node.stream == "existing"

    if old_config is not None:
        sys.modules["ml_switcheroo_compiler.core.config"] = old_config
