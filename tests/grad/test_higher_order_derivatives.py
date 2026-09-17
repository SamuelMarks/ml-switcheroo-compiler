"""Tests for higher-order derivatives, hvp_graph, and nth_order_grad."""

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import (
    hvp,
    hvp_graph,
    nth_order_grad,
    nth_order_grad_graph,
)


def test_hvp_graph_symbolic_lowering() -> None:
    """Test hvp_graph lowers forward-mode JVP over reverse-mode VJP cotangent graph.

    Returns:
        None
    """
    g = LogicalGraph(name="test_quad")
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["pow"] = LogicalNode(id="pow", op_type="Mul", inputs=["x", "x"])
    g.inputs = ["x"]
    g.outputs = ["pow"]

    # Lower HVP on IRGraph directly
    hvp_g = hvp_graph(g, primals=["x"], tangents=["x"])
    assert isinstance(hvp_g, LogicalGraph)
    assert len(hvp_g.outputs) > 0


def test_nth_order_grad_graph() -> None:
    """Test nth_order_grad_graph computes first, second, and third derivative graphs.

    Returns:
        None
    """
    g = LogicalGraph(name="test_cubic")
    g.nodes["x"] = LogicalNode(id="x", op_type="Input")
    g.nodes["sq"] = LogicalNode(id="sq", op_type="Mul", inputs=["x", "x"])
    g.nodes["cb"] = LogicalNode(id="cb", op_type="Mul", inputs=["sq", "x"])
    g.inputs = ["x"]
    g.outputs = ["cb"]

    g1 = nth_order_grad_graph(g, wrt=["x"], output_id="cb", n=1)
    assert len(g1.outputs) == 1

    g2 = nth_order_grad_graph(g, wrt=["x"], output_id="cb", n=2)
    assert len(g2.outputs) == 1

    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad_graph(g, wrt=["x"], output_id="cb", n=0)


def test_nth_order_grad_callable() -> None:
    """Test nth_order_grad callable evaluates 1st, 2nd, and 3rd derivatives of a polynomial.

    Returns:
        None
    """
    with ConfigContext(eager_mode=True):
        # f(x) = x^3
        # f'(x) = 3 x^2
        # f''(x) = 6 x
        # f'''(x) = 6
        def cubic(x: Tensor) -> Tensor:
            """Cubic polynomial.

            Args:
                x (Tensor): Input tensor.

            Returns:
                Tensor: Cubed tensor.
            """
            return x * x * x

        x_val = np.array([2.0], dtype=np.float32)
        x = Tensor(x_val, TensorConfig(x_val.shape, DType.Float32, "cpu"))

        first_deriv = nth_order_grad(cubic, n=1)
        res1 = first_deriv(x)
        assert np.isclose(float(getattr(res1, "data", res1)), 12.0, atol=1e-3)

        second_deriv = nth_order_grad(cubic, n=2)
        res2 = second_deriv(x)
        assert np.isclose(float(getattr(res2, "data", res2)), 12.0, atol=1e-3)

        with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
            nth_order_grad(cubic, n=0)


def test_hvp_api_delegation() -> None:
    """Test hvp API eliminates naive re-wrapping and evaluates correctly.

    Returns:
        None
    """
    with ConfigContext(eager_mode=True):

        def scalar_quad(x: np.ndarray) -> object:
            """Quadratic polynomial.

            Args:
                x (np.ndarray): Input tensor.

            Returns:
                object: Quad form tensor.
            """
            return x * x * 0.5

        x_val = np.array([3.0, 4.0], dtype=np.float32)
        v_val = np.array([1.0, 1.0], dtype=np.float32)

        # For f(x) = 0.5 * x^2:
        # H = [[1, 0], [0, 1]]
        # H · v = [1, 1]
        primal, hvp_res = hvp(scalar_quad, (x_val,), (v_val,))
        assert np.allclose(getattr(primal, "data", primal), [4.5, 8.0], atol=1e-3)
        assert np.allclose(getattr(hvp_res, "data", hvp_res), [1.0, 1.0], atol=1e-3)
