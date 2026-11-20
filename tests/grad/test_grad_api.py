"""Unit tests for `ml_switcheroo_compiler.grad.api` differentiation modes and symbolic wrappers."""

from __future__ import annotations

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
    ir_grad,
    nth_order_grad,
    nth_order_grad_graph,
    overwrite_with_gradient,
    value_and_grad,
)
from ml_switcheroo_compiler.grad.options import GradOptions
from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import ensure_tensor, unpack_pair


def _build_cubic_graph(include_tangent: bool = False) -> LogicalGraph:
    """Construct a symbolic computation graph computing `x3 = (x * x) * x`.

    Args:
        include_tangent (bool): Whether to include an input tangent node `v` for HVP evaluation.

    Returns:
        LogicalGraph: Symbolic computation graph with scalar output `x3`.
    """
    graph = LogicalGraph(name="cubic_graph")
    nodes: dict[str, LogicalNode] = {
        "x": LogicalNode(id="x", op_type="Input", inputs=[], attributes={}, shape_metadata=()),
        "x2": LogicalNode(id="x2", op_type="Mul", inputs=["x", "x"], attributes={}, shape_metadata=()),
        "x3": LogicalNode(id="x3", op_type="Mul", inputs=["x2", "x"], attributes={}, shape_metadata=()),
    }
    graph.inputs = ["x"]
    if include_tangent:
        nodes["v"] = LogicalNode(id="v", op_type="Input", inputs=[], attributes={}, shape_metadata=())
        graph.inputs.append("v")
    graph.nodes = NodeDict(nodes)
    graph.outputs = ["x3"]
    return graph


def test_grad_api_overwrite_with_gradient() -> None:
    """Verify `overwrite_with_gradient` preserves forward values and overrides backward cotangents."""
    cfg = TensorConfig((1,), DType.Float32, Device("cpu"))
    t = Tensor(np.array([2.0], dtype=np.float32), cfg)
    g = Tensor(np.array([5.0], dtype=np.float32), cfg)

    out = overwrite_with_gradient(t, g)
    np.testing.assert_allclose(np.asarray(ensure_tensor(out).numpy()), [2.0])

    def target_fn(x: Tensor) -> Tensor:
        """Scale an overwritten-gradient tensor by 2.0.

        Args:
            x (Tensor): Primal input tensor.

        Returns:
            Tensor: Scaled primal output tensor.
        """
        overwritten = overwrite_with_gradient(x, g)
        return ensure_tensor(overwritten) * 2.0

    vag = value_and_grad(target_fn)
    val, computed_grad = unpack_pair(vag(t))
    np.testing.assert_allclose(np.asarray(ensure_tensor(val).numpy()), [4.0])
    np.testing.assert_allclose(np.asarray(computed_grad), 5.0)


def test_grad_api_backward_non_proxy_data_and_empty_argnums() -> None:
    """Verify `backward()` with raw non-proxy scalar data and `grad()` with empty `argnums`."""
    graph = LogicalGraph(name="non_proxy_loss_graph")
    node = LogicalNode(id="1.0", op_type="Constant", inputs=[], attributes={"value": 1.0}, shape_metadata=())
    graph.add_node(node)

    loss_tensor = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))

    global_tracing_state.start_tracing("non_proxy_loss_graph")
    global_tracing_state.active_graph = graph
    try:
        backward(loss_tensor)
        assert loss_tensor.grad == 1.0
    finally:
        global_tracing_state.stop_tracing()

    empty_grad_fn = grad(lambda x: x * 3.0, options=GradOptions(argnums=()))
    empty_grads = empty_grad_fn(2.0)
    assert empty_grads == ()


def test_grad_api_grad_and_ir_grad_modes() -> None:
    """Verify `grad`, `ir_grad`, and `value_and_grad` across `has_aux=True` and `has_aux=False`."""
    opts_aux = GradOptions(has_aux=True)

    def fn_aux(x: float) -> tuple[float, str]:
        """Compute scaled output and auxiliary string metadata.

        Args:
            x (float): Scalar input value.

        Returns:
            tuple[float, str]: Tuple of `(5 * x, 'auxiliary')`.
        """
        return x * 5.0, "auxiliary"

    grad_aux_fn = grad(fn_aux, options=opts_aux)
    grads, aux = unpack_pair(grad_aux_fn(2.0))
    assert np.isclose(float(np.asarray(grads)), 5.0)
    assert aux == "auxiliary"

    vag_aux_fn = value_and_grad(fn_aux, options=opts_aux)
    loss_and_aux, vag_grad = unpack_pair(vag_aux_fn(4.0))
    loss, vag_aux = unpack_pair(loss_and_aux)
    assert np.isclose(float(np.asarray(loss)), 20.0)
    assert vag_aux == "auxiliary"
    assert np.isclose(float(np.asarray(vag_grad)), 5.0)

    ir_grad_fn = ir_grad(lambda x: x * x)
    assert np.isclose(float(np.asarray(ir_grad_fn(3.0))), 6.0)


def test_grad_api_nth_order_grad_graph_and_callable() -> None:
    """Verify `nth_order_grad_graph` and `nth_order_grad` compute exact higher-order derivatives."""
    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad_graph(LogicalGraph(), ["x"], "out", n=0)

    with pytest.raises(ValueError, match="Derivative order n must be >= 1"):
        nth_order_grad(lambda x: x, n=0)

    cubic_graph = _build_cubic_graph()
    second_deriv_graph = nth_order_grad_graph(cubic_graph, ["x"], "x3", n=2)
    eval_out = evaluate_graph(second_deriv_graph, {"x": np.array(3.0, dtype=np.float32)})
    out_id = second_deriv_graph.outputs[0]
    assert np.isclose(float(np.asarray(eval_out[out_id])), 18.0)

    def cubic(x: float) -> float:
        """Evaluate cubic polynomial `x^3`.

        Args:
            x (float): Scalar input value.

        Returns:
            float: Evaluated `x^3`.
        """
        return (x * x) * x

    first_order = nth_order_grad(cubic, n=1)
    second_order = nth_order_grad(cubic, n=2)
    assert np.isclose(float(np.asarray(first_order(3.0))), 27.0)
    assert np.isclose(float(np.asarray(second_order(3.0))), 18.0)


def test_grad_api_hvp_graph_and_hvp() -> None:
    """Verify `hvp_graph` (default and explicit outputs) and `hvp` evaluate exact Hessian-vector products."""
    cubic_graph_default = _build_cubic_graph(include_tangent=True)
    sym_hvp_default = hvp_graph(cubic_graph_default, ["x"], ["v"], outputs=None)
    inputs = {"x": np.array(3.0, dtype=np.float32), "v": np.array(2.0, dtype=np.float32)}
    res_default = evaluate_graph(sym_hvp_default, dict(inputs))
    assert np.isclose(float(np.asarray(res_default[sym_hvp_default.outputs[-1]])), 36.0)

    cubic_graph_explicit = _build_cubic_graph(include_tangent=True)
    sym_hvp_explicit = hvp_graph(cubic_graph_explicit, ["x"], ["v"], outputs=["x3"])
    res_explicit = evaluate_graph(sym_hvp_explicit, dict(inputs))
    assert np.isclose(float(np.asarray(res_explicit[sym_hvp_explicit.outputs[-1]])), 36.0)

    primal_val, hvp_val = hvp(lambda x: (x * x) * x, 3.0, 2.0, has_aux=False)
    assert np.isclose(float(np.asarray(primal_val)), 27.0)
    assert np.isclose(float(np.asarray(hvp_val)), 36.0)
