"""Tests for test_grad_api_coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import (
    backward,
    grad,
    hvp,
    hvp_graph,
    nth_order_grad,
    nth_order_grad_graph,
    overwrite_with_gradient,
    value_and_grad,
)
from ml_switcheroo_compiler.grad.options import GradOptions
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_grad_api_overwrite_with_gradient_full() -> None:
    """Verify overwrite_with_gradient execution and custom VJP registration."""
    # Test forward pass with concrete tensor
    cfg = TensorConfig((1,), DType.Float32, Device("cpu"))
    t = Tensor(np.array([2.0], dtype=np.float32), cfg)
    g = Tensor(np.array([5.0], dtype=np.float32), cfg)

    # Calling overwrite_with_gradient invokes _overwrite and wraps fwd and bwd
    out = overwrite_with_gradient(t, g)
    assert out is not None

    # Test backward logic of overwrite_with_gradient directly via custom VJP invocation
    def target_fn(x: Tensor) -> Tensor:
        """Function with gradient overwrite."""
        overwritten = overwrite_with_gradient(x, g)
        return overwritten * 2.0

    # Under value_and_grad
    vag = value_and_grad(target_fn)
    val, computed_grad = vag(t)
    assert val is not None
    assert computed_grad is not None


def test_grad_api_backward_non_proxy_data() -> None:
    """Verify backward() when tensor.data has no .id attribute (loss_id fallback to str(data))."""
    graph = LogicalGraph(name="test_graph")
    node = LogicalNode(id="raw_val_str", op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=())
    graph.add_node(node)

    # Tensor with raw non-proxy data whose str is 'raw_val_str'
    class NonProxyData:
        """Dummy data object without .id attribute."""

        def __str__(self) -> str:
            """String representation."""
            return "raw_val_str"

    loss_tensor = Tensor(NonProxyData(), TensorConfig((), DType.Float32, Device("cpu")))

    global_tracing_state.start_tracing("test_graph")
    global_tracing_state.active_graph = graph
    try:
        with patch("ml_switcheroo_compiler.grad.api._find_wrt_tensors", return_value=([], [])):
            backward(loss_tensor)
            assert loss_tensor.grad == 1.0
    finally:
        global_tracing_state.stop_tracing()


def test_grad_api_grad_has_aux_branch() -> None:
    """Verify grad() function when options.has_aux=True."""
    opts = GradOptions(has_aux=True)

    def fn_aux(x: float) -> tuple[float, str]:
        """Primal function with auxiliary payload."""
        return x * 5.0, "auxiliary"

    grad_fn = grad(fn_aux, options=opts)
    grads, aux = grad_fn(2.0)
    assert np.isclose(grads, 5.0)
    assert aux == "auxiliary"


def test_grad_api_value_and_grad_has_aux_branch() -> None:
    """Verify value_and_grad with has_aux=True branches in api.py."""
    opts = GradOptions(has_aux=True)

    def fn_with_aux(x: float) -> tuple[float, str]:
        """Primal function with auxiliary payload."""
        return x * 3.0, "aux_info"

    vag_fn = value_and_grad(fn_with_aux, options=opts)
    (loss, aux), grad_val = vag_fn(4.0)
    assert aux == "aux_info"
    assert loss == 12.0
    assert np.isclose(grad_val, 3.0)


def test_grad_api_nth_order_grad_graph() -> None:
    """Verify nth_order_grad_graph calculation and validation."""
    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad_graph(LogicalGraph(), ["x"], "out", n=0)

    # Valid graph differentiation
    graph = LogicalGraph(name="poly_graph")
    graph.inputs = ["x"]
    graph.outputs = ["x"]
    graph.nodes = {
        "x": LogicalNode(id="x", op_type="Input", inputs=[], attributes={}, shape_metadata=()),
    }

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_g1 = LogicalGraph(name="g1")
        mock_g1.outputs = ["out1"]
        mock_g2 = LogicalGraph(name="g2")
        mock_g2.outputs = ["out2"]
        mock_grad.side_effect = [mock_g1, mock_g2]

        res = nth_order_grad_graph(graph, ["x"], "x", n=2)
        assert res is mock_g2
        assert mock_grad.call_count == 2


def test_grad_api_nth_order_grad() -> None:
    """Verify nth_order_grad higher order wrapper function."""
    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad(lambda x: x, n=0)

    # Verify chaining
    f = lambda x: x * x * x
    first_order = nth_order_grad(f, n=1)
    res_1 = first_order(2.0)
    assert np.isclose(res_1, 12.0)


def test_grad_api_symbolic_hvp_and_hvp() -> None:
    """Verify hvp_graph and hvp wrappers."""
    graph = LogicalGraph(name="hvp_graph")
    graph.outputs = ["out_id"]

    with patch("ml_switcheroo_compiler.transforms.autodiff.hvp") as mock_hvp:
        mock_hvp.return_value = "symbolic_hvp_graph"
        # Test with outputs=None
        res1 = hvp_graph(graph, ["x"], ["v"], outputs=None)
        assert res1 == "symbolic_hvp_graph"

        # Test with explicit outputs
        res2 = hvp_graph(graph, ["x"], ["v"], outputs=["custom_out"])
        assert res2 == "symbolic_hvp_graph"

    # Test top-level hvp function
    with patch("ml_switcheroo_compiler.grad.jvp_vjp.hvp", return_value=("prim_val", "hvp_val")) as mock_jvp_hvp:
        out = hvp(lambda x: x * x, 2.0, 1.0, has_aux=False)
        assert out == ("prim_val", "hvp_val")
        mock_jvp_hvp.assert_called_once()
