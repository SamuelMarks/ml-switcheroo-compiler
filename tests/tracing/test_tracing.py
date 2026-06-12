"""Unit tests for the tracer tape and proxy tensor components of the ML Switcheroo tracing.

system

This module verifies the behavior of tracing contexts, mathematical operations on proxy
tensors, error handling outside active tracing contexts, and AST reference
propagation.
"""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo.tracing.tracer import ProxyTensor, TracerTape, _tracer


def test_tracer_tape() -> None:
    """Verifies the lifecycle and state transitions of the TracerTape.

    This test ensures that nodes cannot be added when tracing is inactive,
    that starting tracing initializes a graph with the correct name, that
    nodes are successfully added to the active graph, and that stopping
    tracing correctly returns the constructed graph and resets the tracing state

    Returns:
    None.
    """
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
    """Verifies mathematical and matrix multiplication operations on ProxyTensor objects.

    This test checks element-wise operations (addition, subtraction, multiplication,
    division, exponentiation), right-side operations with scalars, matrix
    multiplication shape propagation, and error handling for invalid matrix
    multiplication operands within an active tracing context

    Returns:
    None.
    """
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
    """Verifies that performing operations on ProxyTensor objects outside an active.

    tracing

    context raises a RuntimeError

    This ensures that tracing operations are strictly bound to active tracer tape
    sessions

    Returns:
    None.
    """
    a = ProxyTensor(id="a", shape=(2, 3))
    b = ProxyTensor(id="b", shape=(2, 3))

    with pytest.raises(RuntimeError):
        _ = a + b

    with pytest.raises(RuntimeError):
        _ = a @ b


def test_tracer_add_node_with_ast_ref() -> None:
    """Verifies that AST references are correctly preserved when adding logical nodes to.

    the tracer tape

    This test ensures that metadata such as source AST references are successfully
    propagated through the tracing process and stored in the final logical graph

    Returns:
    None.
    """
    tape = TracerTape()
    tape.start_tracing("Test")
    n = LogicalNode(id="n1", op_type="Input", source_ast_ref="test:1")
    tape.add_node(n)
    out_graph = tape.stop_tracing()
    assert out_graph.nodes["n1"].source_ast_ref == "test:1"
