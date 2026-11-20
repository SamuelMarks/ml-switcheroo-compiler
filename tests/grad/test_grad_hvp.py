"""Tests for test_grad_hvp."""

from __future__ import annotations

import numpy as np
import pytest
from ml_switcheroo_ir import NodeDict

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import check_numerical_grads, hvp


def test_hvp_robustness() -> None:
    """Test function test_hvp_robustness."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return (x * x) * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, out_tan = hvp(f, x, v)
    assert ensure_ndarray(val).item() == 8.0
    assert ensure_ndarray(out_tan).item() == 12.0  # 6 * x * v = 6 * 2 * 1 = 12


def test_hessian() -> None:
    """Test function test_hessian."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return (x * x) * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    from ml_switcheroo_compiler.grad import hessian

    res = hessian(f)(x)
    np.testing.assert_allclose(np.asarray(res), [[12.0]])


def test_check_numerical_grads() -> None:
    """Test function test_check_numerical_grads."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x (Tensor): Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return x * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    check_numerical_grads(f, (x,))
    assert x.shape == (1,)


def test_grad_of_grad() -> None:
    """Test function test_grad_of_grad."""
    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.transforms.autodiff import grad

    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input")
    n_mul1 = IRNode(id="mul1", op_type="Multiply", inputs=["x", "x"])
    n_mul2 = IRNode(id="mul2", op_type="Multiply", inputs=["mul1", "x"])
    g.nodes = NodeDict({"x": n_in, "mul1": n_mul1, "mul2": n_mul2})
    g.inputs = ["x"]
    g.outputs = ["mul2"]

    grad_g = grad(g, ["x"], "mul2")
    # Take grad of grad with respect to x again
    grad_grad_g = grad(grad_g, ["x"], grad_g.outputs[0])
    assert "x" in [n.id for n in grad_grad_g.nodes.values() if n.op_type == "Input"]
    assert len(grad_grad_g.outputs) == 1


def _removed() -> None:
    """Test function _removed."""
    from unittest.mock import patch

    # the function we are calling is hvp_graph (which is hvp inside autodiff)
    # it uses has_jvp and has_vjp from the module scope of autodiff!
    # Let's bypass has_jvp and has_vjp by replacing them entirely!
    from ml_switcheroo_ir import IRGraph, LogicalNode

    from ml_switcheroo_compiler.transforms.autodiff import hvp as hvp_graph

    g = IRGraph()
    n_in = LogicalNode(id="x", op_type="Input")
    n_if = LogicalNode(id="cond", op_type="Loop", inputs=["x"])
    g.nodes = NodeDict({"x": n_in, "cond": n_if})
    g.inputs = ["x"]
    g.outputs = ["cond"]

    # We must patch `has_jvp` and `has_vjp` inside `ml_switcheroo_compiler.transforms.autodiff` BEFORE `hvp` calls them!
    with patch("ml_switcheroo_compiler.transforms.autodiff.has_jvp", return_value=True, create=True):
        with patch("ml_switcheroo_compiler.transforms.autodiff.has_vjp", return_value=True, create=True):
            with pytest.raises(NotImplementedError):
                hvp_graph(g, ["x"], ["v"], ["cond"])
