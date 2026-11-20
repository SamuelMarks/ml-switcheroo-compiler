"""Unit tests for the graph interpreter and runtime execution environment."""

from __future__ import annotations

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.interpreter import evaluate_graph
from ml_switcheroo_compiler.interpreter.environment import Environment


def test_evaluate_graph() -> None:
    """Verify `evaluate_graph` computes Input -> Add(Constant) -> Relu accurately."""
    g = LogicalGraph(outputs=["out"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["c"] = LogicalNode(id="c", op_type="Constant", attributes={"value": 2.0})
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in", "c"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Relu", inputs=["add"])
    outputs = evaluate_graph(g, {"in": np.array([-3.0, 1.0])})
    assert "out" in outputs
    np.testing.assert_allclose(ensure_ndarray(outputs["out"]), np.array([0.0, 3.0]))


def test_missing_input() -> None:
    """Verify `evaluate_graph` raises an error when a required graph input is missing."""
    g = LogicalGraph(outputs=["in"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    with pytest.raises((ValueError, KeyError, ShapeMismatchError)):
        evaluate_graph(g, {})


def test_not_implemented() -> None:
    """Verify `evaluate_graph` raises an error for an unregistered operation type."""
    g = LogicalGraph(outputs=["out"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="UnknownOp", inputs=["in"])
    with pytest.raises(Exception, match=r".*"):
        evaluate_graph(g, {"in": np.array([1.0])})


def test_missing_output() -> None:
    """Verify `evaluate_graph` raises `RuntimeError` when a declared graph output is never evaluated."""
    g = LogicalGraph(outputs=["missing"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    with pytest.raises(RuntimeError, match="never evaluated"):
        evaluate_graph(g, {"in": np.array([1.0])})


def test_all_ops() -> None:
    """Verify `evaluate_graph` evaluates composite arithmetic, reduction, matrix, and broadcast ops."""
    g = LogicalGraph(outputs=["out"])
    g.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    g.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    g.nodes["sub"] = LogicalNode(id="sub", op_type="Sub", inputs=["in1", "in2"])
    g.nodes["mul"] = LogicalNode(id="mul", op_type="Mul", inputs=["sub", "in1"])
    g.nodes["div"] = LogicalNode(id="div", op_type="Div", inputs=["mul", "in2"])
    g.nodes["neg"] = LogicalNode(id="neg", op_type="Neg", inputs=["div"])
    g.nodes["exp"] = LogicalNode(id="exp", op_type="Exp", inputs=["neg"])
    g.nodes["log"] = LogicalNode(id="log", op_type="Log", inputs=["exp"])
    g.nodes["pow"] = LogicalNode(id="pow", op_type="Pow", inputs=["log", "in2"])
    g.nodes["sum"] = LogicalNode(id="sum", op_type="Sum", inputs=["pow"])
    g.nodes["mean"] = LogicalNode(id="mean", op_type="Mean", inputs=["sum"])
    g.nodes["max"] = LogicalNode(id="max", op_type="Max", inputs=["pow"])
    g.nodes["min"] = LogicalNode(id="min", op_type="Min", inputs=["pow"])
    g.nodes["in3"] = LogicalNode(id="in3", op_type="Input")
    g.nodes["trans"] = LogicalNode(id="trans", op_type="Transpose", inputs=["in3"])
    g.nodes["matmul"] = LogicalNode(id="matmul", op_type="MatMul", inputs=["in3", "trans"])
    g.nodes["greater"] = LogicalNode(id="greater", op_type="Greater", inputs=["in1", "in2"])
    g.nodes["expand"] = LogicalNode(id="expand", op_type="Expand", inputs=["min"], shape_metadata=(2,))
    g.nodes["out"] = LogicalNode(id="out", op_type="Add", inputs=["expand", "expand"])
    inputs = {
        "in1": np.array([2.0, 3.0]),
        "in2": np.array([1.0, 1.0]),
        "in3": np.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    res = evaluate_graph(g, inputs)
    assert "out" in res
    np.testing.assert_allclose(ensure_ndarray(res["out"]), np.array([-12.0, -12.0]))


def test_interpreter_environment_state_management() -> None:
    """Verify `Environment` initialization, lookup, mutation, containment, and missing-key error."""
    env_default = Environment()
    assert env_default.memory == {}

    env = Environment({"var_x": 3.14, "dim_n": 128})
    assert "var_x" in env
    assert "missing_key" not in env
    assert env.get("var_x") == 3.14

    with pytest.raises(ValueError, match="Missing input value for node 'not_found'"):
        env.get("not_found")

    env.set("var_y", 42)
    assert env.get("var_y") == 42
    assert "var_y" in env
