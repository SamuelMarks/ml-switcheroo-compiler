"""Tests for LogPoissonLoss autodiff VJP/JVP rules and numerical gradient verification."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from ml_switcheroo_ir import LogicalNode

from ml_switcheroo_compiler.backends.numpy.eager.nn_polyfills import (
    _np_log_poisson_loss,
    _stirling_gammaln,
)
from ml_switcheroo_compiler.ir.core import IRGraph
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    log_poisson_loss_jvp,
    log_poisson_loss_vjp,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_stirling_gammaln_accuracy() -> None:
    """Test Stirling approximation of gammaln against known factorial values."""
    # Gamma(1) = 0! = 1 => ln Gamma(1) = 0.0
    val_1 = float(_stirling_gammaln(np.array([1.0]))[0])
    assert abs(val_1) < 1e-3

    # Gamma(5) = 4! = 24 => ln(24) = 3.1780538
    val_5 = float(_stirling_gammaln(np.array([5.0]))[0])
    assert abs(val_5 - np.log(24.0)) < 1e-4

    # Vectorized evaluation
    arr = np.array([2.0, 3.0, 4.0, 10.0])
    res = _stirling_gammaln(arr)
    assert res.shape == (4,)
    assert np.all(np.isfinite(res))


def test_log_poisson_loss_eager_modes() -> None:
    """Test LogPoissonLoss eager numerical calculation across log and non-log modes."""
    targets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    predictions = np.array([0.5, 1.0, 1.5], dtype=np.float32)

    # log_input=True: exp(pred) - targets * pred
    loss_log = _np_log_poisson_loss(np, targets, predictions, log_input=True)
    expected_log = np.exp(predictions) - targets * predictions
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(loss_log, expected_log, rtol=1e-5)

    # log_input=False: pred - targets * log(pred)
    loss_nonlog = _np_log_poisson_loss(np, targets, predictions, log_input=False)
    expected_nonlog = predictions - targets * np.log(predictions)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(loss_nonlog, expected_nonlog, rtol=1e-5)

    # compute_full_loss=True
    loss_full = _np_log_poisson_loss(np, targets, predictions, log_input=True, compute_full_loss=True)
    assert cast("Callable[..., EagerTensorProtocol]", np.all)(cast("EagerTensorProtocol", loss_full) >= cast("EagerTensorProtocol", loss_log))


def test_log_poisson_loss_vjp_rule() -> None:
    """Test LogPoissonLoss VJP symbolic rule generation."""
    graph = IRGraph()
    node = LogicalNode(
        id="loss_1",
        op_type="LogPoissonLoss",
        inputs=["targets", "log_inputs"],
        attributes={"log_input": True},
        shape_metadata=(4,),
    )
    graph.nodes["loss_1"] = node

    vjp_ids = log_poisson_loss_vjp(graph, node, "cotangent_1")
    assert len(vjp_ids) == 2
    target_grad_id, pred_grad_id = vjp_ids

    assert target_grad_id in graph.nodes
    assert pred_grad_id in graph.nodes
    assert cast("EagerTensorProtocol", getattr(graph.nodes, "__getitem__")(target_grad_id)).op_type == "ZerosLike"
    assert cast("EagerTensorProtocol", getattr(graph.nodes, "__getitem__")(pred_grad_id)).op_type == "Mul"


def test_log_poisson_loss_vjp_nonlog_rule() -> None:
    """Test LogPoissonLoss VJP symbolic rule with log_input=False."""
    graph = IRGraph()
    node = LogicalNode(
        id="loss_2",
        op_type="LogPoissonLoss",
        inputs=["targets", "inputs"],
        attributes={"log_input": False},
        shape_metadata=(4,),
    )
    graph.nodes["loss_2"] = node

    vjp_ids = log_poisson_loss_vjp(graph, node, "cotangent_2")
    assert len(vjp_ids) == 2
    assert vjp_ids[1] in graph.nodes
    assert cast("EagerTensorProtocol", getattr(graph.nodes, "__getitem__")(cast("EagerTensorProtocol", getattr(vjp_ids, "__getitem__")(1)))).op_type == "Mul"


def test_log_poisson_loss_jvp_rule() -> None:
    """Test LogPoissonLoss JVP forward derivative rule generation."""
    graph = IRGraph()
    node = LogicalNode(
        id="loss_1",
        op_type="LogPoissonLoss",
        inputs=["targets", "log_inputs"],
        attributes={"log_input": True},
        shape_metadata=(4,),
    )
    graph.nodes["loss_1"] = node

    jvp_id = log_poisson_loss_jvp(graph, node, ["tan_targets", "tan_log_inputs"])
    assert jvp_id in graph.nodes
    assert cast("EagerTensorProtocol", getattr(graph.nodes, "__getitem__")(jvp_id)).op_type == "Mul"


def test_log_poisson_loss_jvp_nonlog_rule() -> None:
    """Test LogPoissonLoss JVP forward derivative rule generation with log_input=False."""
    graph = IRGraph()
    node = LogicalNode(
        id="loss_2",
        op_type="LogPoissonLoss",
        inputs=["targets", "inputs"],
        attributes={"log_input": False},
        shape_metadata=(4,),
    )
    graph.nodes["loss_2"] = node

    jvp_id = log_poisson_loss_jvp(graph, node, ["tan_targets", "tan_inputs"])
    assert jvp_id in graph.nodes
    assert cast("EagerTensorProtocol", getattr(graph.nodes, "__getitem__")(jvp_id)).op_type == "Mul"


def test_log_poisson_loss_numerical_gradcheck() -> None:
    """Verify analytical LogPoissonLoss gradient against central finite differences."""
    targets = np.array([2.0, 3.0, 1.0], dtype=np.float64)
    log_input = np.array([0.7, 1.2, -0.3], dtype=np.float64)
    eps = 1e-5

    # Analytical gradient w.r.t log_input: exp(log_input) - targets
    analytical_grad = np.exp(log_input) - targets

    # Finite difference gradient
    numerical_grad = np.zeros_like(log_input)
    for i in range(len(log_input)):
        plus = log_input.copy()
        minus = log_input.copy()
        plus[i] += eps
        minus[i] -= eps
        l_plus = cast("Callable[..., EagerTensorProtocol]", np.sum)(cast("Callable[..., EagerTensorProtocol]", _np_log_poisson_loss)(np, targets, plus, log_input=True))
        l_minus = cast("Callable[..., EagerTensorProtocol]", np.sum)(cast("Callable[..., EagerTensorProtocol]", _np_log_poisson_loss)(np, targets, minus, log_input=True))
        numerical_grad[i] = (l_plus - l_minus) / (2.0 * eps)

    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-4)
