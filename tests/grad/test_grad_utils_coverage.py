"""Tests for test_grad_utils_coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from ml_switcheroo_ir import LogicalGraph

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig, Variable
from ml_switcheroo_compiler.grad.options import GradOptions
from ml_switcheroo_compiler.grad.utils import (
    _compute_grad_and_value,
    _find_wrt_tensors,
    _to_original_type,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_grad_utils_find_wrt_tensors_branches() -> None:
    """Verify _find_wrt_tensors branch coverage when tensors have variable requirements."""
    graph = LogicalGraph(name="test_graph")
    graph.nodes = {"node_1": MagicMock(), "node_2": MagicMock(), "node_3": MagicMock()}

    proxy1 = ProxyTensor(id="node_1", shape=(), dtype="float32")
    t1 = Tensor(proxy1, TensorConfig((), DType.Float32, Device("cpu")))
    t1._requires_grad = True

    proxy2 = ProxyTensor(id="node_2", shape=(), dtype="float32")
    t2 = Tensor(proxy2, TensorConfig((), DType.Float32, Device("cpu")))
    t2.trainable = True

    proxy3 = ProxyTensor(id="node_3", shape=(), dtype="float32")
    t3 = Variable(proxy3, TensorConfig((), DType.Float32, Device("cpu")))

    # An unrelated tensor
    proxy_unrelated = ProxyTensor(id="unrelated_node", shape=(), dtype="float32")
    t_unrelated = Tensor(proxy_unrelated, TensorConfig((), DType.Float32, Device("cpu")))
    t_unrelated._requires_grad = True

    # Tensor without .id on data
    t_no_id = Tensor("raw_string", TensorConfig((), DType.Float32, Device("cpu")))

    # Tensor not in graph.nodes
    proxy_not_in_graph = ProxyTensor(id="node_not_in_graph", shape=(), dtype="float32")
    t_not_in_graph = Tensor(proxy_not_in_graph, TensorConfig((), DType.Float32, Device("cpu")))
    t_not_in_graph._requires_grad = True

    # Tensor in graph with no requires_grad/trainable/Variable
    proxy_no_grad = ProxyTensor(id="node_1", shape=(), dtype="float32")
    t_no_grad = Tensor(proxy_no_grad, TensorConfig((), DType.Float32, Device("cpu")))

    # Mock gc.get_objects to return these specific tensors
    with patch("gc.get_objects", return_value=[t1, t2, t3, t_unrelated, t_no_id, t_not_in_graph, t_no_grad, "not_a_tensor"]):
        wrt_tensors, wrt_ids = _find_wrt_tensors(graph)
        assert any(t is t1 for t in wrt_tensors)
        assert any(t is t2 for t in wrt_tensors)
        assert any(t is t3 for t in wrt_tensors)
        assert not any(t is t_unrelated for t in wrt_tensors)
        assert not any(t is t_no_id for t in wrt_tensors)
        assert not any(t is t_not_in_graph for t in wrt_tensors)
        assert not any(t is t_no_grad for t in wrt_tensors)
        assert "node_1" in wrt_ids
        assert "node_2" in wrt_ids
        assert "node_3" in wrt_ids


def test_grad_utils_to_original_type_branches() -> None:
    """Verify _to_original_type branches when converting back to tensors."""
    cfg = TensorConfig((2,), DType.Float32, Device("cpu"))
    orig_tensor = Tensor(np.array([1.0, 2.0], dtype=np.float32), cfg)

    # 1. val is already a Tensor
    already_tensor = Tensor(np.array([3.0, 4.0], dtype=np.float32), cfg)
    res1 = _to_original_type(already_tensor, orig_tensor)
    assert res1 is already_tensor

    # 2. bool array to Tensor
    bool_arr = np.array([True, False], dtype=np.bool_)
    res2 = _to_original_type(bool_arr, orig_tensor)
    assert isinstance(res2, Tensor)
    assert res2.dtype == DType.Bool

    # 3. int array to Tensor
    int_arr = np.array([1, 2], dtype=np.int32)
    res3 = _to_original_type(int_arr, orig_tensor)
    assert isinstance(res3, Tensor)
    assert res3.dtype == DType.Int32


def test_grad_utils_compute_grad_and_value_tracing_branch() -> None:
    """Verify _compute_grad_and_value under global_tracing_state with active graph."""
    graph = LogicalGraph(name="trace_compute_graph")
    graph.nodes = {}

    def f_linear(x: Tensor) -> Tensor:
        """Identity function."""
        return x

    proxy = ProxyTensor(id="in_1", shape=(2,), dtype="float32")
    t_in = Tensor(proxy, TensorConfig((2,), DType.Float32, Device("cpu")))

    global_tracing_state.start_tracing("trace_compute_graph")
    global_tracing_state.active_graph = graph
    try:
        opts = GradOptions()
        # Mock vjp inside _compute_grad_and_value
        vjp_fn_mock = MagicMock(return_value=(Tensor(np.array([1.0, 1.0]), TensorConfig((2,), DType.Float32, Device("cpu"))),))
        with patch("ml_switcheroo_compiler.grad.jvp_vjp.vjp", return_value=(t_in, vjp_fn_mock)):
            val, res_grad = _compute_grad_and_value(f_linear, opts, (t_in,))
            assert val is t_in
            assert res_grad is not None
            assert any(node.op_type == "Constant" for node in graph.nodes.values())
    finally:
        global_tracing_state.stop_tracing()
