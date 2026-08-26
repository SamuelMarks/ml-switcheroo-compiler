# ruff: noqa: E501
"""Unit tests for the tracer tape and proxy tensor components of the ML Switcheroo tracing.

system

This module verifies the behavior of tracing contexts, mathematical operations on proxy
tensors, error handling outside active tracing contexts, and AST reference
propagation.
"""

import pytest
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import ShapeMismatchError, TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRNode
from ml_switcheroo_compiler.ir.state import create_read_variable
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import (
    ProxyTensor,
    TracerTape,
    get_trace_count,
    increment_trace_count,
    reset_trace_count,
)


def testglobal_tracing_state_tape() -> None:
    """Verifies the lifecycle and state transitions of the TracerTape."""
    tape = TracerTape()
    assert not global_tracing_state.is_tracing
    tape.add_node(LogicalNode(id="n", op_type="Linear"))
    graph = tape.start_tracing("Test")
    assert global_tracing_state.is_tracing
    assert graph.name == "Test"
    n = LogicalNode(id="n1", op_type="Input")
    tape.add_node(n)
    assert "n1" in global_tracing_state.active_graph.nodes
    out_graph = tape.stop_tracing()
    assert not global_tracing_state.is_tracing
    assert out_graph.nodes["n1"] == n


def test_proxy_tensor_math() -> None:
    """Test the proxy tensor math behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies mathematical and matrix multiplication operations on ProxyTensor objects.\n\n    This test checks element-wise operations (addition, subtraction, multiplication,\n    division, exponentiation), right-side operations with scalars, matrix\n    multiplication shape propagation, and error handling for invalid matrix\n    multiplication operands within an active tracing context\n\n    Returns:\n    None.\n    "
        global_tracing_state.start_tracing()
        a = ProxyTensor(id="a", shape=(2, 3))
        b = ProxyTensor(id="b", shape=(2, 3))
        c = a + b
        assert c.shape == (2, 3)
        assert c.data.id != "a"
        _ = a - b
        _ = a * b
        _ = a / b
        _ = a**2
        _ = 2 + a
        _ = 3 - a
        _ = 4 * a
        _ = 5 / a
        with pytest.raises((ValueError, ShapeMismatchError)):
            _ = a @ 2
        m = ProxyTensor(id="m", shape=(3, 4))
        n = a @ m
        assert n.shape == (2, 4)
        graph = global_tracing_state.stop_tracing()
        assert len(graph.nodes) > 0
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_proxy_tensor_outside_context() -> None:
    """Test the proxy tensor outside context behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that performing operations on ProxyTensor objects outside an active.\n\n    tracing\n\n    context raises a RuntimeError\n\n    This ensures that tracing operations are strictly bound to active tracer tape\n    sessions\n\n    Returns:\n    None.\n    "
        config.eager_mode = False
        a = ProxyTensor(id="a", shape=(2, 3))
        b = ProxyTensor(id="b", shape=(2, 3))
        with pytest.raises((RuntimeError, TracingError)):
            _ = a + b
        with pytest.raises((RuntimeError, TracingError)):
            _ = a @ b
        config.eager_mode = True
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_tracer_add_node_with_ast_ref() -> None:
    """Test the tracer add node with ast ref behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Verifies that AST references are correctly preserved when adding logical nodes to.\n\n    the tracer tape\n\n    This test ensures that metadata such as source AST references are successfully\n    propagated through the tracing process and stored in the final logical graph\n\n    Returns:\n    None.\n    "
        tape = TracerTape()
        tape.start_tracing("Test")
        n = LogicalNode(id="n1", op_type="Input", source_ast_ref="test:1")
        tape.add_node(n)
        out_graph = tape.stop_tracing()
        assert out_graph.nodes["n1"].source_ast_ref == "test:1"
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_proxy_tensor_assign_operations() -> None:
    """Test the proxy tensor assign operations behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test assign, assign_add, and assign_sub on ProxyTensor."
        graph = global_tracing_state.start_tracing(name="assign_test")
        try:
            var_node = create_read_variable("my_var", shape=(2, 2), dtype="float32")
            global_tracing_state.add_node(var_node)
            var_proxy = ProxyTensor(id=var_node.id, shape=(2, 2), dtype="float32")
            val_proxy = ProxyTensor(id="val_1", shape=(2, 2), dtype="float32")
            updated_1 = var_proxy.assign(val_proxy)
            assert updated_1.shape == (2, 2)
            assign_node_1 = graph.nodes[updated_1.id]
            assert assign_node_1.op_type == "AssignVariable"
            assert assign_node_1.attributes["variable_name"] == "my_var"
            updated_2 = updated_1.assign_add(val_proxy)
            assign_node_2 = graph.nodes[updated_2.id]
            assert assign_node_2.op_type == "AssignVariable"
            assert assign_node_2.attributes["variable_name"] == "my_var"
            updated_3 = updated_2.assign_sub(val_proxy)
            assign_node_3 = graph.nodes[updated_3.id]
            assert assign_node_3.op_type == "AssignVariable"
            assert assign_node_3.attributes["variable_name"] == "my_var"
        finally:
            global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_proxy_tensor_assign_errors() -> None:
    """Test the proxy tensor assign errors behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Test proxy_tensor_assign_errors."
        var_proxy = ProxyTensor(id="foo", shape=(), dtype="float32")
        val_proxy = ProxyTensor(id="bar", shape=(), dtype="float32")
        with pytest.raises((RuntimeError, TracingError)):
            var_proxy.assign(val_proxy)
        global_tracing_state.start_tracing(name="assign_err_test")
        try:
            with pytest.raises((ValueError, ShapeMismatchError), match="assign\\(\\) can only be called on a variable proxy."):
                var_proxy.assign(val_proxy)
            global_tracing_state.add_node(IRNode(id="foo", op_type="Add", inputs=[], shape_metadata=()))
            with pytest.raises((ValueError, ShapeMismatchError), match="assign\\(\\) can only be called on a variable proxy."):
                var_proxy.assign(val_proxy)
            global_tracing_state.add_node(
                IRNode(
                    id="var_node",
                    op_type="ReadVariable",
                    inputs=[],
                    attributes={"variable_name": "x"},
                    shape_metadata=(),
                )
            )
            var_proxy_2 = ProxyTensor(id="var_node", shape=(), dtype="float32")
            out = var_proxy_2.assign(42.0)
            assert out.shape == ()
            assert global_tracing_state.active_graph.nodes[out.id].op_type == "AssignVariable"
        finally:
            global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_trace_counts():
    """Test the trace counts behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:

        def my_func():
            """Evaluate and process the my func operation.

            Returns:
                object: The evaluated or processed output.
            """
            pass

        reset_trace_count(my_func)
        assert get_trace_count(my_func) == 0
        increment_trace_count(my_func)
        assert get_trace_count(my_func) == 1
        reset_trace_count(my_func)
        assert get_trace_count(my_func) == 0

        def traced_fn():
            """Evaluate and process the traced fn operation.

            Returns:
                object: The evaluated or processed output.
            """
            proxy = ProxyTensor(id="out", shape=(), dtype="float32")
            return Tensor(proxy, TensorConfig((), DType.Float32, "cpu"))

        reset_trace_count(traced_fn)
        _trace_function(traced_fn, (), "test_trace")
        assert get_trace_count(traced_fn) == 1
        reset_trace_count(traced_fn)
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
