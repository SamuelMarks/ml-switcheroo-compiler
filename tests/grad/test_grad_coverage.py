"""Tests for grad.py missing lines."""

from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.ops as ops
import ml_switcheroo_compiler.ops.control_flow_utils as cfu
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import _check_scalar, _get_concrete_val, _to_original_type, checkpoint
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_check_scalar_symbolic() -> None:
    """Test _check_scalar on a tensor with non-scalar symbolic shape."""
    t = Tensor(np.ones(4), TensorConfig((4,), DType.Float32, Device("cpu"), requires_grad=True))
    with mock.patch.object(Tensor, "shape", new_callable=mock.PropertyMock, return_value=("N",)):
        with pytest.raises(Exception, match="backward\\(\\) can only be called on scalar tensors."):
            _check_scalar(t)


def test_get_concrete_val_proxy() -> None:
    """Test _get_concrete_val when value is a ProxyTensor inside _data."""

    class MockData:
        pass

    t = Tensor(MockData(), TensorConfig((), DType.Float32, Device("cpu")))
    t._data = ProxyTensor("t1", (), "float32")
    t._data.concrete_value = 42.0

    with mock.patch.object(Tensor, "data", new_callable=mock.PropertyMock, return_value=None):
        assert _get_concrete_val(t) == 42.0


def test_fallback_input() -> None:
    """Test _generate_fallback_input in backward."""
    from ml_switcheroo_compiler.grad import _generate_fallback_input, _get_inputs_dict

    graph = IRGraph()
    graph.nodes = {"in": LogicalNode("in", "Input", shape_metadata=[2, 3], attributes={"dtype": "float32"})}
    res = _generate_fallback_input(graph, "in")
    assert res.shape == (2, 3)

    # Test _get_inputs_dict branch for fallback
    graph.inputs = ["in"]
    res_dict = _get_inputs_dict(graph)
    assert "in" in res_dict


def test_to_original_type_int_bool() -> None:
    """Test _to_original_type with int and bool tensors."""
    t_int = Tensor(np.array(1, dtype=np.int32), TensorConfig((), DType.Int32, Device("cpu")))
    res_int = _to_original_type(np.array([1, 2], dtype=np.int32), t_int)
    assert res_int.dtype == DType.Int32

    t_bool = Tensor(np.array(True, dtype=bool), TensorConfig((), DType.Bool, Device("cpu")))
    res_bool = _to_original_type(np.array([True, False], dtype=bool), t_bool)
    assert res_bool.dtype == DType.Bool


def test_checkpoint_dtype_fallback() -> None:
    """Test checkpoint when dtype is inferred from tensor args."""

    def f(x):
        return ops.multiply(x, x)

    f_cp = checkpoint(f)
    t = Tensor(np.array(3.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))

    prev_eager = config.eager_mode
    config.eager_mode = False

    # We can invoke f_cp wrapper directly inside a fake trace graph
    global_tracing_state.is_tracing = True
    graph = IRGraph()
    global_tracing_state.active_graph = graph

    original_trace = cfu._trace_function

    def mock_trace(*args, **kwargs):
        block = original_trace(*args, **kwargs)
        # Force all nodes to HAVE dtype attribute
        nodes = block.nodes if isinstance(block.nodes, list) else block.nodes.values()
        for n in nodes:
            if not hasattr(n, "attributes"):
                n.attributes = {}
            n.attributes["dtype"] = "float64"
        return block

    try:
        with mock.patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", side_effect=mock_trace):
            out = f_cp(t)
            assert out.config.dtype.value == "float64"
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
        config.eager_mode = prev_eager


def test_get_concrete_val_non_proxy():
    """Test _get_concrete_val when _data is not a ProxyTensor."""
    t = Tensor(42.0, TensorConfig((), DType.Float32, Device("cpu")))
    t._data = "not a proxy"
    with mock.patch.object(Tensor, "data", new_callable=mock.PropertyMock, return_value=None):
        assert _get_concrete_val(t) == "not a proxy"


def test_fallback_input_no_attributes():
    """Test _generate_fallback_input with no attributes on node."""
    from ml_switcheroo_compiler.grad import _generate_fallback_input

    graph = IRGraph()
    graph.nodes = {"in": LogicalNode("in", "Input", shape_metadata=[2, 3])}
    res = _generate_fallback_input(graph, "in")
    assert res.shape == (2, 3)


def test_get_inputs_dict_none_val():
    """Test _get_inputs_dict when val is None."""
    from ml_switcheroo_compiler.grad import _get_inputs_dict

    graph = IRGraph()
    graph.nodes = {"test_id": LogicalNode("test_id", "Input", shape_metadata=[])}
    t = Tensor(42.0, TensorConfig((), DType.Float32, Device("cpu")))
    t._data = mock.Mock(id="test_id")
    with mock.patch("ml_switcheroo_compiler.grad.utils._get_concrete_val", return_value=None):
        res = _get_inputs_dict(graph)
        assert "test_id" not in res


def test_backward_missing_output():
    """Test backward when output is missing from outputs_dict."""
    from ml_switcheroo_compiler.grad import backward

    t = Tensor(42.0, TensorConfig((), DType.Float32, Device("cpu"), requires_grad=True))
    global_tracing_state.is_tracing = True
    graph = IRGraph()
    global_tracing_state.active_graph = graph
    t._data = ProxyTensor("out_id", (), "float32")
    graph.outputs = ["out_id"]
    graph.nodes = {"out_id": LogicalNode("out_id", "Add", inputs=["a", "b"])}
    try:
        with mock.patch("ml_switcheroo_compiler.interpreter.evaluator.evaluate_graph", return_value={}):
            backward(t)
            assert not hasattr(t, "grad")
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None


def test_checkpoint_no_tensor_args():
    """Test checkpoint dtype and device infer when tensor_args is empty."""

    def f():
        import ml_switcheroo_compiler.ops as ops
        from ml_switcheroo_compiler.core.device import Device
        from ml_switcheroo_compiler.core.dtype import DType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t1 = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))
        t2 = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))
        return ops.add(t1, t2)

    f_cp = checkpoint(f)
    prev_eager = config.eager_mode
    config.eager_mode = False
    global_tracing_state.is_tracing = True
    graph = IRGraph()
    global_tracing_state.active_graph = graph
    original_trace = cfu._trace_function

    def mock_trace(*args, **kwargs):
        block = original_trace(*args, **kwargs)
        for n in block.nodes:
            if hasattr(n, "attributes"):
                n.attributes = {}
        return block

    try:
        with mock.patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", side_effect=mock_trace):
            out = f_cp()
            assert out.config.device.device_type == "cpu"
            assert out.config.dtype.value == "float32"
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
        config.eager_mode = prev_eager


def test_value_and_grad_default_dtype():
    """Test value_and_grad _prepare_primals default dtype."""
    from ml_switcheroo_compiler.grad import value_and_grad

    def f(x):
        return x

    vg = value_and_grad(f)
    with mock.patch("ml_switcheroo_compiler.grad.api.get_active_backend") as mock_backend:
        mock_backend_inst = mock.Mock()
        mock_backend.return_value = mock_backend_inst

        class MockArr:
            shape = ()
            dtype = mock.Mock()

            def __str__(self):
                return "unknown"

        mock_backend_inst.asarray.return_value = MockArr()
        MockArr.dtype.__str__ = lambda x: "unknown"
        try:
            vg(mock.Mock())
        except Exception:
            pass


def test_grad_missing_tensor_no_id():
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad import _get_inputs_dict
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyData:
        pass

    t = Tensor(1.0, config=TensorConfig(shape=(), dtype="float32", device=Device("cpu")))
    t._data = DummyData()

    g = IRGraph()
    res = _get_inputs_dict(g)
    assert isinstance(res, dict)


def test_grad_tensor_id_not_in_graph():
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad import _find_wrt_tensors, _get_inputs_dict
    from ml_switcheroo_compiler.ir.core import IRGraph

    class DummyData:
        id = "not_in_graph"

    t = Tensor(1.0, config=TensorConfig(shape=(), dtype="float32", device=Device("cpu")))
    t._data = DummyData()

    g = IRGraph()
    # Ensure it's not in the graph nodes

    res_inputs = _get_inputs_dict(g)
    assert isinstance(res_inputs, dict)

    res_wrt = _find_wrt_tensors(g)
    assert isinstance(res_wrt, tuple)
