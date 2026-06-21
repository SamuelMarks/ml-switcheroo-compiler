"""Unit tests for the tracer tape and proxy tensor components of the ML Switcheroo tracing.

system

This module verifies the behavior of tracing contexts, mathematical operations on proxy
tensors, error handling outside active tracing contexts, and AST reference
propagation.
"""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, TracerTape, _tracer


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


def test_proxy_tensor_assign_operations() -> None:
    """Test assign, assign_add, and assign_sub on ProxyTensor."""
    from ml_switcheroo_compiler.ir.state import create_read_variable
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    graph = _tracer.start_tracing(name="assign_test")
    try:
        # Manually create a variable read to act as our variable proxy
        var_node = create_read_variable("my_var", shape=(2, 2), dtype="float32")
        _tracer.add_node(var_node)
        var_proxy = ProxyTensor(id=var_node.id, shape=(2, 2), dtype="float32")

        # Test assign
        val_proxy = ProxyTensor(id="val_1", shape=(2, 2), dtype="float32")
        updated_1 = var_proxy.assign(val_proxy)
        assert updated_1.shape == (2, 2)
        assign_node_1 = graph.nodes[updated_1.id]
        assert assign_node_1.op_type == "AssignVariable"
        assert assign_node_1.attributes["variable_name"] == "my_var"

        # Test assign_add
        updated_2 = updated_1.assign_add(val_proxy)
        assign_node_2 = graph.nodes[updated_2.id]
        assert assign_node_2.op_type == "AssignVariable"
        assert assign_node_2.attributes["variable_name"] == "my_var"

        # Test assign_sub
        updated_3 = updated_2.assign_sub(val_proxy)
        assign_node_3 = graph.nodes[updated_3.id]
        assert assign_node_3.op_type == "AssignVariable"
        assert assign_node_3.attributes["variable_name"] == "my_var"

    finally:
        _tracer.stop_tracing()


def test_proxy_tensor_assign_errors() -> None:
    """Test proxy_tensor_assign_errors."""
    import pytest

    from ml_switcheroo_compiler.ir.core import IRNode
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    var_proxy = ProxyTensor(id="foo", shape=(), dtype="float32")
    val_proxy = ProxyTensor(id="bar", shape=(), dtype="float32")

    # Test outside tracing
    with pytest.raises(RuntimeError, match="Cannot perform assign outside of a tracing context."):
        var_proxy.assign(val_proxy)

    _tracer.start_tracing(name="assign_err_test")
    try:
        # Test assign on non-existent node
        with pytest.raises(ValueError, match=r"assign\(\) can only be called on a variable proxy."):
            var_proxy.assign(val_proxy)

        # Test assign on non-variable node
        _tracer.add_node(IRNode(id="foo", op_type="Add", inputs=[], shape_metadata=()))
        with pytest.raises(ValueError, match=r"assign\(\) can only be called on a variable proxy."):
            var_proxy.assign(val_proxy)

        # Test assigning a constant value (not a ProxyTensor)
        _tracer.add_node(
            IRNode(
                id="var_node",
                op_type="ReadVariable",
                inputs=[],
                attributes={"variable_name": "x"},
                shape_metadata=(),
            ),
        )
        var_proxy_2 = ProxyTensor(id="var_node", shape=(), dtype="float32")
        out = var_proxy_2.assign(42.0)
        assert out.shape == ()
        assert _tracer.active_graph.nodes[out.id].op_type == "AssignVariable"

    finally:
        _tracer.stop_tracing()


def test_trace_counts():
    from ml_switcheroo_compiler.tracing.tracer import (
        get_trace_count,
        increment_trace_count,
        reset_trace_count,
    )

    def my_func():
        pass

    reset_trace_count(my_func)
    assert get_trace_count(my_func) == 0
    increment_trace_count(my_func)
    assert get_trace_count(my_func) == 1
    reset_trace_count(my_func)
    assert get_trace_count(my_func) == 0

    # Test increment through _trace_function
    from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function

    def traced_fn():
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

        proxy = ProxyTensor(id="out", shape=(), dtype="float32")
        return Tensor(proxy, TensorConfig((), DType.Float32, "cpu"))

    reset_trace_count(traced_fn)
    _trace_function(traced_fn, (), "test_trace")
    assert get_trace_count(traced_fn) == 1
    reset_trace_count(traced_fn)
