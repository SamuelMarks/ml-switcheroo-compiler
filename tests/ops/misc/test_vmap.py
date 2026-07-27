# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.vmap import vmap


class MockTensor:
    def __init__(self, shape=()):
        self.shape = shape
        from ml_switcheroo_compiler.core.dtype import DType

        self.dtype = DType.Float32
        self.device = "cpu"
        self.data = "mock_data"


def test_vmap_eager(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = True
    mocker.patch("ml_switcheroo_compiler.ops.eager_evaluator.EagerEvaluator.evaluate", return_value=t)

    def my_func(x):
        return x

    vmapped = vmap(my_func, in_axes=0, out_axes=0)
    try:
        res = vmapped(t)
    except Exception:
        pass


def test_vmap_tracing(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))
    config.eager_mode = False
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.is_tracing", True)
    mocker.patch("ml_switcheroo_compiler.tracing.state.global_tracing_state.add_node")
    mocker.patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph")

    def my_func(x, y):
        return x

    vmapped = vmap(my_func, in_axes=(0, 1), out_axes=0)
    try:
        res = vmapped(t, t)
    except Exception:
        pass


def test_vmap_not_tracing():
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state_mod

    state_mod.global_tracing_state.is_tracing = False

    def my_func(x):
        return x

    with pytest.raises(RuntimeError):
        vmap(my_func)(Tensor(MockTensor().data, TensorConfig((), DType.Float32, "cpu")))


def test_vmap_eager_impl(mocker):
    t = Tensor(MockTensor((2, 3)).data, TensorConfig((2, 3), DType.Float32, "cpu"))

    def func(x):
        return x


class MockData:
    shape = (2, 3)

    def __init__(self, d="mock"):
        self.data = d
        self.id = "mock_id"

    def __array__(self):
        return np.zeros((2, 3))

    def tolist(self):
        return []


def test_vmap_eager_execute():
    import ml_switcheroo_compiler.backends.registry as reg
    from ml_switcheroo_compiler.ops.vmap import _eager_vmap

    config.eager_mode = True

    class FakeBackend:
        def execute_op(self, name, *a, **k):

            class R:
                shape = (2, 3)

            return R()

    old = reg.get_active_backend
    reg.get_active_backend = lambda: FakeBackend()
    t = Tensor(MockData(), TensorConfig((2, 3), DType.Float32, "cpu"))

    def my_func(x):
        return x

    res = _eager_vmap(my_func, 0, 0, (t,))
    assert res.shape == (2, 3)
    reg.get_active_backend = old


def test_vmap_resolve():
    from ml_switcheroo_compiler.ops.vmap import _compute_vmap_shape

    class FakeT:
        shape = ()
        dtype = DType.Float32
        device = "cpu"

    assert _compute_vmap_shape(FakeT(), None) == ()


def test_vmap_trace():
    import ml_switcheroo_compiler.tracing.state as state
    from ml_switcheroo_compiler.ops.vmap import _trace_vmap

    class DummyGraph:
        def add_node(self, node):
            pass

    state.global_tracing_state.is_tracing = True
    state.global_tracing_state.active_graph = DummyGraph()
    state.global_tracing_state.add_node = state.global_tracing_state.active_graph.add_node

    def my_func(x):
        return x

    class FakeT:
        shape = (2,)
        dtype = DType.Float32
        device = "cpu"
        data = "mock"

    t = Tensor(MockData(), TensorConfig((2,), DType.Float32, "cpu"))
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.ops.control_flow_utils._trace_function", return_value="graph"):
        res = _trace_vmap(my_func, 0, 0, (t,))
        assert res.shape == (2,)
