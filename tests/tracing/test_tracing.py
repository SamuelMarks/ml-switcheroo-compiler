"""Module docstring."""

import pytest
from ml_switcheroo.tracing import TracerTape, ProxyTensor, _tracer
from ml_switcheroo_ir import LogicalNode


def test_tracer_tape() -> None:
    """Docstring."""
    tape = TracerTape()
    assert not tape.is_tracing

    with pytest.raises(RuntimeError):
        tape.add_node(LogicalNode(id="n", op_type="Linear"))

    graph = tape.start_tracing("Test")
    assert tape.is_tracing
    assert graph.name == "Test"

    n = LogicalNode(id="n1", op_type="Input")
    tape.add_node(n)
    assert "n1" in tape.active_graph.nodes

    out_graph = tape.stop_tracing()
    assert not tape.is_tracing
    assert out_graph.nodes["n1"] == n


def test_proxy_tensor_math() -> None:
    """Docstring."""
    _tracer.start_tracing()

    a = ProxyTensor(id="a", shape=(2, 3))
    b = ProxyTensor(id="b", shape=(2, 3))

    c = a + b
    assert c.shape == (2, 3)
    assert c.id != "a"

    _ = a - b
    _ = a * b
    _ = a / b
    _ = a**2

    # Right-side math
    _ = 2 + a
    _ = 3 - a
    _ = 4 * a
    _ = 5 / a

    with pytest.raises(ValueError):
        _ = a @ 2

    m = ProxyTensor(id="m", shape=(3, 4))
    n = a @ m
    assert n.shape == (2, 4)

    graph = _tracer.stop_tracing()
    assert len(graph.nodes) > 0


def test_proxy_tensor_outside_context() -> None:
    """Docstring."""
    a = ProxyTensor(id="a", shape=(2, 3))
    b = ProxyTensor(id="b", shape=(2, 3))

    with pytest.raises(RuntimeError):
        _ = a + b

    with pytest.raises(RuntimeError):
        _ = a @ b


def test_tracer_add_node_with_ast_ref() -> None:
    """Docstring."""
    tape = TracerTape()
    tape.start_tracing("Test")
    n = LogicalNode(id="n1", op_type="Input", source_ast_ref="test:1")
    tape.add_node(n)
    out_graph = tape.stop_tracing()
    assert out_graph.nodes["n1"].source_ast_ref == "test:1"
