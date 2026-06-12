"""Tests for autodiff and evaluation error handling in ml_switcheroo_compiler.

This module defines a mock operator with an unimplemented VJP to verify that the
autodiff engine correctly raises errors when attempting to differentiate unsupported
operations. It also tests input validation errors during graph evaluation.
"""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.ops.base import OpDef, register_op
from ml_switcheroo.transforms.autodiff import grad


@register_op("TestUnimplVjp")
class TestUnimplVjp(OpDef):
    """A mock operator definition used to test unimplemented VJP behavior.

    This operator registers under the name 'TestUnimplVjp' and explicitly
    raises a NotImplementedError when its vector-jacobian product (VJP)
    is requested, allowing verification of error handling in the autodiff system.
    """

    def vjp(self, graph: object, node: object, cotangent: object) -> None:
        """Vjp function."""
        msg = "Not implemented"
        raise NotImplementedError(msg)

    def jvp(self, *args: object, **kwargs: object) -> None:
        """Jvp function."""

    def infer_shape(self, *args: object, **kwargs: object) -> None:
        """infer_shape function."""

    def numpy_eval(self, *args: object, **kwargs: object) -> None:
        """numpy_eval function."""

    def emit_jax(self, *args: object, **kwargs: object) -> None:
        """emit_jax function."""

    def emit_keras(self, *args: object, **kwargs: object) -> None:
        """emit_keras function."""

    def emit_mlx(self, *args: object, **kwargs: object) -> None:
        """emit_mlx function."""

    def emit_pytorch(self, *args: object, **kwargs: object) -> None:
        """emit_pytorch function."""

    def emit_tensorflow(self, *args: object, **kwargs: object) -> None:
        """emit_tensorflow function."""


def test_grad_unimpl_vjp() -> None:
    """Tests that calculating the gradient of an unimplemented VJP raises an error.

    This test constructs a logical graph containing the 'TestUnimplVjp' operator
    and asserts that attempting to compute its gradient raises an exception
    indicating that the VJP is not implemented

    Returns:
    None
    """
    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    g.nodes["b"] = LogicalNode(id="b", op_type="TestUnimplVjp", inputs=["a"])
    with pytest.raises(Exception, match="VJP not implemented for"):
        grad(g, ["a"], "b")


def test_evaluator_dict_error() -> None:
    """Tests that evaluating a graph with missing inputs raises a ValueError.

    This test constructs a logical graph with an input node and attempts to
    evaluate it with an empty inputs dictionary, verifying that the evaluator
    correctly detects and reports the missing input

    Returns:
    None
    """
    from ml_switcheroo.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    # Missing input
    with pytest.raises(ValueError):
        evaluate_graph(g, inputs={})
