from unittest.mock import MagicMock, patch

from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

"""Tests for tracing control flow."""

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.tracing import cond_tracing, map_fn_tracing, scan_tracing, while_loop_tracing


class MockData:
    def __init__(self, id_val="mock"):
        self.id = id_val


@pytest.fixture(autouse=True)
def enable_tracing_mode():
    config.eager_mode = False
    global_tracing_state.start_tracing()
    yield
    global_tracing_state.stop_tracing()


def test_cond_tracing():
    pred = Tensor(MockData("pred"), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn():
        return Tensor(MockData("t"), TensorConfig((), DType.Float32, Device("cpu")))

    def false_fn():
        return Tensor(MockData("f"), TensorConfig((), DType.Float32, Device("cpu")))

    res = cond_tracing(pred, true_fn, false_fn)
    assert res.config.dtype == DType.Float32


def test_while_loop_tracing():
    def cond_fn(x):
        return Tensor(MockData("c"), TensorConfig((), DType.Bool, Device("cpu")))

    def body_fn(x):
        return [Tensor(MockData("b"), TensorConfig((), DType.Float32, Device("cpu")))]

    init_val = Tensor(MockData("init"), TensorConfig((), DType.Float32, Device("cpu")))
    res = while_loop_tracing(cond_fn, body_fn, [init_val])
    assert res[0].config.dtype == DType.Float32


def test_scan_tracing():
    def scan_fn(carry, x):
        new_carry = Tensor(MockData("nc"), TensorConfig((), DType.Float32, Device("cpu")))
        return [new_carry], [new_carry]

    init = [Tensor(MockData("init"), TensorConfig((), DType.Float32, Device("cpu")))]
    xs = Tensor(MockData("xs"), TensorConfig((3,), DType.Float32, Device("cpu")))

    res_carry, res_ys = scan_tracing(scan_fn, init, xs)
    assert res_carry[0].config.dtype == DType.Float32
    assert res_ys[0].config.dtype == DType.Float32


def test_map_fn_tracing():
    def fn(x):
        return Tensor(MockData("m"), TensorConfig((), DType.Float32, Device("cpu")))

    elems = Tensor(MockData("elems"), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = map_fn_tracing(fn, elems)
    assert res.config.dtype == DType.Float32


def test_pmap_tracing():
    from ml_switcheroo_compiler.ops.control_flow.tracing import pmap_tracing

    @pmap_tracing
    def fn(x):
        return Tensor(MockData("p"), TensorConfig((), DType.Float32, Device("cpu")))

    x = Tensor(MockData("x"), TensorConfig((), DType.Float32, Device("cpu")))
    res = fn(x)
    assert res.config.dtype == DType.Float32


def test_stop_gradient_tracing():
    from ml_switcheroo_compiler.ops.control_flow.tracing import stop_gradient_tracing

    x = Tensor(MockData("x"), TensorConfig((), DType.Float32, Device("cpu")))
    res = stop_gradient_tracing(x)
    assert res.config.dtype == DType.Float32
    assert res.config.requires_grad is False


def test_assert_value_tracing():
    from ml_switcheroo_compiler.ops.control_flow.tracing import assert_value_tracing

    x = Tensor(MockData("x"), TensorConfig((), DType.Bool, Device("cpu")))
    assert_value_tracing(x, "test msg")


import pytest

from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.ops.control_flow.tracing import _flatten_inputs, assert_value_tracing, pmap_tracing, stop_gradient_tracing


def test_tracing_error_outside_context():
    global_tracing_state.is_tracing = False
    with pytest.raises(TracingError):
        cond_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        while_loop_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        scan_tracing(MagicMock(), MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        map_fn_tracing(MagicMock(), MagicMock())
    with pytest.raises(TracingError):
        pmap_tracing(MagicMock())(1)

    assert stop_gradient_tracing(1) == 1

    with patch("ml_switcheroo_compiler.ops.control_flow.tracing.record_assertion") as mock_assert:
        assert_value_tracing(MagicMock(), "msg")
        mock_assert.assert_called_once()


def test_while_loop_tensor_init():
    def c(x):
        return x

    def b(x):
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))

    global_tracing_state.start_tracing()
    res = while_loop_tracing(c, b, t)
    global_tracing_state.stop_tracing()
    assert isinstance(res, Tensor)


def test_flatten_inputs_other():
    assert _flatten_inputs(1) == []


def test_pmap_tracing_other_args():
    def f(x, y):
        return x

    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    global_tracing_state.start_tracing()
    res = pmap_tracing(f)(t, 1)
    global_tracing_state.stop_tracing()


def test_stop_gradient_tracing_traced():
    t = Tensor(ProxyTensor("t", (), DType.Int32.value), TensorConfig((), DType.Int32, None))
    p = ProxyTensor("p", (), DType.Int32.value)

    global_tracing_state.start_tracing()
    res1 = stop_gradient_tracing(t)
    assert isinstance(res1, Tensor)

    res2 = stop_gradient_tracing(p)
    assert isinstance(res2, ProxyTensor)
    global_tracing_state.stop_tracing()
