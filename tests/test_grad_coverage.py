"""Tests for autodiff and evaluation error handling in ml_switcheroo_compiler.

This module defines a mock operator with an unimplemented VJP to verify that the
autodiff engine correctly raises errors when attempting to differentiate unsupported
operations. It also tests input validation errors during graph evaluation.
"""

import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.tensor import TensorConfig
from ml_switcheroo_compiler.ops.base import OpDef, register_op
from ml_switcheroo_compiler.transforms.autodiff import grad


@register_op("TestUnimplVjp")
class TestUnimplVjp(OpDef):
    """A mock operator definition used to test unimplemented VJP behavior.

    This operator registers under the name 'TestUnimplVjp' and explicitly
    raises a NotImplementedError when its vector-jacobian product (VJP)
    is requested, allowing verification of error handling in the autodiff system.
    """

    def vjp(self, graph: object, node: object, cotangent: object) -> None:
        """Vjp function.

        Args:
            graph (object): The graph.
            node (object): The node.
            cotangent (object): The cotangent.
        """
        msg = "Not implemented"
        raise NotImplementedError(msg)

    def jvp(self, *args: object, **kwargs: object) -> None:
        """Jvp function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def infer_shape(self, *args: object, **kwargs: object) -> None:
        """infer_shape function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def eager_eval(self, *args: object, **kwargs: object) -> None:
        """eager_eval function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def emit_jax(self, *args: object, **kwargs: object) -> None:
        """emit_jax function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def emit_keras(self, *args: object, **kwargs: object) -> None:
        """emit_keras function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def emit_mlx(self, *args: object, **kwargs: object) -> None:
        """emit_mlx function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def emit_pytorch(self, *args: object, **kwargs: object) -> None:
        """emit_pytorch function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """

    def emit_tensorflow(self, *args: object, **kwargs: object) -> None:
        """emit_tensorflow function.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """


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
    from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

    g = LogicalGraph()
    g.nodes["a"] = LogicalNode(id="a", op_type="Input")
    # Missing input
    with pytest.raises(ValueError):
        evaluate_graph(g, inputs={})


def test_grad_frontend_mocks() -> None:
    """Test the mock frontend functions in grad.py."""
    from ml_switcheroo_compiler.grad import (
        backward,
        custom_jvp,
        custom_vjp,
        disable_jit,
        eval_shape,
    )
    from ml_switcheroo_compiler.grad import grad as frontend_grad
    from ml_switcheroo_compiler.grad import (
        ir_grad,
        jit,
        jvp,
        value_and_grad,
        value_and_grad_wrt_vars,
        vjp,
    )

    def my_fun(x: int) -> int:
        """Docstring."""
        return x * 2

    # test ir_grad and grad
    assert ir_grad(my_fun)(5) == 10
    assert frontend_grad(my_fun)(5) == 10

    # test value_and_grad
    val, g = value_and_grad(my_fun)(5)
    assert val == 10
    assert g == 10

    # test value_and_grad_wrt_vars
    val_v, g_v = value_and_grad_wrt_vars(my_fun)(5)
    assert val_v == 10
    assert g_v == {}

    # test jit
    assert jit(my_fun)(5) == 10

    # test disable_jit
    with disable_jit():
        assert my_fun(5) == 10

    # test eval_shape
    assert eval_shape(my_fun, 5) == 10

    # test jvp
    out_primal, out_tangent = jvp(my_fun, [5], [1])
    assert out_primal == 10
    assert out_tangent == [1]

    # test vjp
    out_primal, vjp_fn = vjp(my_fun, 5)
    assert out_primal == 10
    assert vjp_fn(2) == (2,)

    # test custom_vjp
    assert custom_vjp(my_fun)(5) == 10

    # test custom_jvp
    assert custom_jvp(my_fun) == my_fun

    # test backward
    assert backward(None) is None


def test_custom_vjp_lazy() -> None:
    """Test custom vjp lazy."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad import custom_vjp
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    @custom_vjp
    def f(x: object) -> object:
        """F."""
        return x * 2  # type: ignore[operator]

    def fwd(x: object) -> object:
        """Fwd."""
        return f(x), x

    def bwd(res: object, g: object) -> object:
        """Bwd."""
        return (res * g,)  # type: ignore[operator]

    f.defvjp(fwd, bwd)

    config.eager_mode = False
    _tracer.start_tracing()

    from unittest.mock import MagicMock

    x = Tensor(MagicMock(id="inp_x"), TensorConfig((2,), DType.Float32, Device("cpu")))
    f(x)

    assert _tracer.active_graph is not None
    assert list(_tracer.active_graph.nodes.values())[-1].op_type == "CustomVJP"

    _tracer.stop_tracing()
    config.eager_mode = True


def test_check_numerical_grads():
    from ml_switcheroo_compiler.grad import check_numerical_grads

    check_numerical_grads(lambda x: x, (1.0,))
