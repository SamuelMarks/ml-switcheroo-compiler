import ast

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.interpreter.evaluator import _evaluate_node, _get_op_alias, _handle_getitem, _handle_slice, _parse_constant, _parse_list, _parse_name, _parse_slice_call, _parse_slice_node, _parse_slice_string, _parse_tuple, _parse_unary, _prepare_node_kwargs, evaluate_graph


def test_evaluator_coverage():
    # evaluate_graph
    g = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="Input", inputs=[]), "n2": LogicalNode(id="n2", op_type="Add", inputs=["n1", "n1"])}, outputs=["n2"])
    inputs = {"n1": np.array(1.0)}
    res = evaluate_graph(g, inputs.copy())
    assert res["n2"] == 2.0

    # missing output
    g_err = LogicalGraph(nodes={"n1": LogicalNode(id="n1", op_type="Input", inputs=[])}, outputs=["n2"])
    with pytest.raises(RuntimeError):
        evaluate_graph(g_err, inputs.copy())

    # _parse_slice_call
    assert _parse_slice_call(ast.parse("slice(1, 2)", mode="eval").body) == slice(1, 2)
    assert _parse_slice_call(ast.parse("array([1, 2])", mode="eval").body) == [1, 2]
    assert _parse_slice_call(ast.parse("unknown(1)", mode="eval").body) is None

    class DummyCall(ast.Call):
        func = ast.Constant(value=1)

    assert _parse_slice_call(DummyCall()) is None

    # _parse_tuple
    assert _parse_tuple(ast.parse("(1, 2)", mode="eval").body) == (1, 2)

    # _parse_list
    assert _parse_list(ast.parse("[1, 2]", mode="eval").body) == [1, 2]

    # _parse_constant
    assert _parse_constant(ast.parse("1", mode="eval").body) == 1

    # _parse_name
    assert _parse_name(ast.Name(id="None")) is None
    assert _parse_name(ast.Name(id="Ellipsis")) is Ellipsis
    assert _parse_name(ast.Name(id="True")) is True

    # _parse_unary
    assert _parse_unary(ast.parse("-1", mode="eval").body) == -1

    class DummyUnary(ast.UnaryOp):
        op = ast.UAdd()
        operand = ast.Constant(value=1)

    assert _parse_unary(DummyUnary()) is None

    # _parse_slice_node
    with pytest.raises(ValueError):
        _parse_slice_node(ast.Pass())

    # _parse_slice_string
    assert _parse_slice_string("slice(1, 2)") == slice(1, 2)

    # _handle_slice
    class EnvMock:
        def __init__(self):
            self.data = {}

        def set(self, k, v):
            self.data[k] = v

        def get(self, k):
            return self.data.get(k)

        def __contains__(self, k):
            return k in self.data

    env = EnvMock()

    class BackendMock:
        def asarray(self, x):
            return np.asarray(x)

        def execute_op(self, op, *args, **kwargs):
            if op == "Meshgrid":
                return [1, 2]
            return 42

        def array(self, x):
            return np.array(x)

    node = LogicalNode(id="slice1", op_type="Slice", inputs=["a"])
    _handle_slice(node, env, BackendMock(), [np.array([1, 2, 3])], {"slices": "slice(0, 1)"})
    assert (env.data["slice1"] == np.array([1])).all()

    _handle_slice(node, env, BackendMock(), [np.array([1, 2, 3])], {"start": 0, "end": 1})
    assert (env.data["slice1"] == np.array([1])).all()

    # _handle_getitem
    _handle_getitem(node, env, BackendMock(), [np.array([1, 2, 3])], {"key": "0"})
    assert env.data["slice1"] == 1

    # _evaluate_node
    # Constant
    n_const = LogicalNode(id="c1", op_type="Constant", attributes={"value": 1})
    _evaluate_node(n_const, env, BackendMock())
    assert env.data["c1"] == 1

    # Slice
    env.set("a", np.array([1, 2]))
    n_slice = LogicalNode(id="s1", op_type="Slice", inputs=["a"], attributes={"start": 0, "end": 1})
    _evaluate_node(n_slice, env, BackendMock())
    assert (env.data["s1"] == np.array([1])).all()

    # GetItem
    n_getitem = LogicalNode(id="g1", op_type="GetItem", inputs=["a"], attributes={"key": "0"})
    _evaluate_node(n_getitem, env, BackendMock())
    assert env.data["g1"] == 1

    # Meshgrid
    n_mesh = LogicalNode(id="m1", op_type="Meshgrid", inputs=["a"], attributes={"output_index": 1})
    _evaluate_node(n_mesh, env, BackendMock())
    assert env.data["m1"] == 2

    # Normal op
    n_add = LogicalNode(id="add1", op_type="Add", inputs=["a"], attributes={})
    _evaluate_node(n_add, env, BackendMock())
    assert env.data["add1"] == 42

    # _get_op_alias
    assert _get_op_alias("Sub") == "Subtract"
    assert _get_op_alias("Unknown") == "Unknown"

    # _prepare_node_kwargs
    n_expand = LogicalNode(id="e1", op_type="Expand", attributes={}, shape_metadata=(1, 2))
    assert _prepare_node_kwargs(n_expand, "Expand") == {"shape": (1, 2)}

    n_reshape = LogicalNode(id="r1", op_type="Reshape", attributes={}, shape_metadata=(1, 2))
    assert _prepare_node_kwargs(n_reshape, "Reshape") == {"newshape": (1, 2)}

    # Check Input again
    n_input = LogicalNode(id="i1", op_type="Input", inputs=[])
    env.set("i1", 100)
    _evaluate_node(n_input, env, BackendMock())
    assert env.data["i1"] == 100
