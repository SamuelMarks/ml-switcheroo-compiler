# ruff: noqa: E501
"""Unit tests for the graph interpreter of the ml_switcheroo_compiler package.

This module contains test cases that verify the correct evaluation of logical graphs,
handling of missing inputs/outputs, unsupported operations, and various mathematical
operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.errors import ShapeMismatchError
from ml_switcheroo_compiler.interpreter import evaluate_graph

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_evaluate_graph() -> None:
    """Test the evaluate graph behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = LogicalGraph(outputs=["out"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["c"] = LogicalNode(id="c", op_type="Constant", attributes={"value": 2.0})
    g.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in", "c"])
    g.nodes["out"] = LogicalNode(id="out", op_type="Relu", inputs=["add"])
    inputs = {"in": np.array([-3.0, 1.0])}
    outputs = cast("EagerCallableProtocol", evaluate_graph)(g, inputs)
    assert "out" in outputs
    cast("EagerCallableProtocol", np.testing.assert_allclose)(outputs["out"], cast("EagerCallableProtocol", np.array)([0.0, 3.0]))


def test_missing_input() -> None:
    """Test the missing input behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = LogicalGraph(outputs=["in"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    with pytest.raises((Exception, ShapeMismatchError)):
        evaluate_graph(g, {})


def test_not_implemented() -> None:
    """Test the not implemented behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = LogicalGraph(outputs=["out"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["out"] = LogicalNode(id="out", op_type="UnknownOp", inputs=["in"])
    with pytest.raises(Exception, match=r".*"):
        cast("EagerCallableProtocol", evaluate_graph)(g, {"in": cast("EagerCallableProtocol", np.array)([1.0])})


def test_missing_output() -> None:
    """Test the missing output behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    g = LogicalGraph(outputs=["missing"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    with pytest.raises(RuntimeError, match="never evaluated"):
        cast("EagerCallableProtocol", evaluate_graph)(g, {"in": cast("EagerCallableProtocol", np.array)([1.0])})


def test_all_ops() -> None:
    """Test the all ops behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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
    inputs = {"in1": np.array([2.0, 3.0]), "in2": np.array([1.0, 1.0]), "in3": np.array([[1.0, 2.0], [3.0, 4.0]])}
    res = cast("EagerCallableProtocol", evaluate_graph)(g, inputs)
    assert "out" in res
    cast("EagerCallableProtocol", np.testing.assert_allclose)(res["out"], cast("EagerCallableProtocol", np.array)([-12.0, -12.0]))
