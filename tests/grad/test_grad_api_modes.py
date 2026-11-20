"""Behavioral tests for higher-order differentiation, auxiliary outputs, and gradient overwrite APIs."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode, NodeDict

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
from tests.testing_utils import (
    MockTestStub,
    coerce_float,
    ensure_tensor,
    extract_test_shape,
    unpack_pair,
)


def test_grad_api_overwrite_with_gradient_forward_and_backward() -> None:
    """Verify `overwrite_with_gradient` preserves primal values and injects custom backward gradients."""
    cfg = TensorConfig((1,), DType.Float32, Device("cpu"))
    t = Tensor(np.array([2.0], dtype=np.float32), cfg)
    g = Tensor(np.array([5.0], dtype=np.float32), cfg)

    out = overwrite_with_gradient(t, g)
    np.testing.assert_allclose(np.asarray(ensure_tensor(out).numpy()), [2.0])

    def target_fn(x: Tensor) -> Tensor:
        """Scale overwritten primal value by two.

        Args:
            x (Tensor): Input primal tensor.

        Returns:
            Tensor: Doubled primal tensor with custom gradient override.
        """
        overwritten = overwrite_with_gradient(x, g)
        return ensure_tensor(overwritten) * 2.0

    vag = value_and_grad(target_fn)
    val, computed_grad = unpack_pair(vag(t))
    np.testing.assert_allclose(np.asarray(ensure_tensor(val).numpy()), [4.0])
    assert tuple(extract_test_shape(computed_grad)) == (1,)
    np.testing.assert_allclose(np.asarray(computed_grad), [5.0])


def test_grad_api_backward_non_proxy_data() -> None:
    """Verify `backward()` resolves `str(tensor.data)` when `tensor.data` lacks `.id` and graph has no parameters."""
    graph = LogicalGraph(name="test_graph")
    node = LogicalNode(id="raw_val_str", op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=())
    graph.add_node(node)

    class _NonProxyData(MockTestStub):
        """Data payload without `.id` attribute."""

        def __str__(self) -> str:
            """Return node identifier string.

            Returns:
                str: Graph node identifier.
            """
            return "raw_val_str"

    loss_tensor = Tensor(
        _NonProxyData(),
        TensorConfig((), DType.Float32, Device("cpu")),
    )

    global_tracing_state.start_tracing("test_graph")
    global_tracing_state.active_graph = graph
    try:
        backward(loss_tensor)
        assert loss_tensor.grad == 1.0
    finally:
        global_tracing_state.stop_tracing()


def test_grad_api_grad_and_value_and_grad_has_aux() -> None:
    """Verify `grad` and `value_and_grad` preserve auxiliary payloads when `has_aux=True`."""
    opts = GradOptions(has_aux=True)

    def fn_aux(x: float) -> tuple[float, str]:
        """Compute `5 * x` with auxiliary tag.

        Args:
            x (float): Scalar input.

        Returns:
            tuple[float, str]: Primal scalar and auxiliary tag.
        """
        return x * 5.0, "auxiliary"

    grad_fn = grad(fn_aux, options=opts)
    grads, aux = unpack_pair(grad_fn(2.0))
    assert np.isclose(coerce_float(grads), 5.0)
    assert aux == "auxiliary"

    vag_fn = value_and_grad(fn_aux, options=opts)
    loss_and_aux, grad_val = unpack_pair(vag_fn(4.0))
    loss, vag_aux = unpack_pair(loss_and_aux)
    assert vag_aux == "auxiliary"
    assert loss == 20.0
    assert np.isclose(coerce_float(grad_val), 5.0)


def test_grad_api_nth_order_grad_and_graph() -> None:
    """Verify `nth_order_grad` and `nth_order_grad_graph` validate order >= 1 and chain derivatives."""
    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad_graph(LogicalGraph(), ["x"], "out", n=0)

    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad(lambda x: x, n=0)

    graph = LogicalGraph(name="poly_graph")
    graph.inputs = ["x"]
    graph.outputs = ["x"]
    graph.nodes = NodeDict(
        {
            "x": LogicalNode(id="x", op_type="Input", inputs=[], attributes={}, shape_metadata=()),
        }
    )

    with patch("ml_switcheroo_compiler.transforms.autodiff.grad") as mock_grad:
        mock_g1 = LogicalGraph(name="g1")
        mock_g1.outputs = ["out1"]
        mock_g2 = LogicalGraph(name="g2")
        mock_g2.outputs = ["out2"]
        mock_grad.side_effect = [mock_g1, mock_g2]

        res = nth_order_grad_graph(graph, ["x"], "x", n=2)
        assert res is mock_g2
        assert mock_grad.call_count == 2

    def cubic(x: float) -> float:
        """Compute `x^3`.

        Args:
            x (float): Scalar input.

        Returns:
            float: Cubic value `x^3`.
        """
        return (x * x) * x

    first_order = nth_order_grad(cubic, n=1)
    res_1 = first_order(2.0)
    assert np.isclose(coerce_float(res_1), 12.0)


def test_grad_api_hvp_and_hvp_graph() -> None:
    """Verify `hvp_graph` default/explicit output forwarding and `hvp` directional second derivative."""
    graph = LogicalGraph(name="hvp_graph")
    graph.inputs = ["x"]
    graph.outputs = ["y"]
    graph.nodes = NodeDict(
        {
            "x": LogicalNode(id="x", op_type="Input", inputs=[], attributes={}, shape_metadata=()),
            "y": LogicalNode(id="y", op_type="Multiply", inputs=["x", "x"], attributes={}, shape_metadata=()),
        }
    )

    g_default = hvp_graph(graph, ["x"], ["v"], outputs=None)
    assert isinstance(g_default, LogicalGraph)
    assert len(g_default.outputs) > 0

    g_custom = hvp_graph(graph, ["x"], ["v"], outputs=["y"])
    assert isinstance(g_custom, LogicalGraph)
    assert len(g_custom.outputs) > 0

    primal_out, hvp_out = hvp(lambda x: x * x * x, 2.0, 1.0, has_aux=False)
    assert np.isclose(coerce_float(primal_out), 8.0)
    assert np.isclose(coerce_float(hvp_out), 12.0, atol=1e-2)
