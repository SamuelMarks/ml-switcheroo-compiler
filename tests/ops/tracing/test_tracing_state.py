# ruff: noqa: E501
import sys

import pytest

from ml_switcheroo_compiler.tracing.state import TracingState


def test_tracing_state():
    state = TracingState()
    assert not state.is_tracing
    graph = state.start_tracing("MyGraph")
    assert state.is_tracing
    assert state.active_graph is graph

    class MockNode:
        def __init__(self):
            self.id = "1"
            self.source_ast_ref = None
            self.domain = ""
            self.stream = None

    node = MockNode()
    state.add_node(node)
    assert node.source_ast_ref is not None
    assert node.domain == "MyGraph"
    if "ml_switcheroo_compiler.core.config" in sys.modules:
        config = sys.modules["ml_switcheroo_compiler.core.config"].config
        old_stream = getattr(config, "current_stream", "default")
        config.current_stream = "my_stream"
        node2 = MockNode()
        node2.id = "2"
        state.add_node(node2)
        assert node2.stream == "my_stream"
        config.current_stream = old_stream
    stopped_graph = state.stop_tracing()
    assert stopped_graph is graph
    assert not state.is_tracing
    with pytest.raises(RuntimeError):
        state.add_node(MockNode())


def test_enrich_ast_and_domain_extra():
    state = TracingState()
    state.start_tracing()

    class MockNode:
        def __init__(self):
            self.source_ast_ref = "existing"
            self.domain = "existing_domain"

    node = MockNode()
    state._enrich_ast_and_domain(node)
    assert node.source_ast_ref == "existing"
    assert node.domain == "existing_domain"
    state.stop_tracing()


def test_enrich_stream_no_config(mocker):
    sys.modules.pop("ml_switcheroo_compiler.core.config", None)
    state = TracingState()

    class MockNode:
        stream = "existing"

    node = MockNode()
    state._enrich_stream(node)
    assert node.stream == "existing"
