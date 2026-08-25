from unittest.mock import MagicMock, patch

"""Tests for eager control flow."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow.eager import cond_eager, while_loop_eager


@pytest.fixture(autouse=True)
def enable_eager_mode():
    config.eager_mode = True
    config.backend = "numpy"
    yield
    config.eager_mode = False
    config.backend = None


def test_cond_eager_true():
    pred = Tensor(np.array(True), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn():
        return Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))

    def false_fn():
        return Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))

    res = cond_eager(pred, true_fn, false_fn)
    assert res.numpy() == 1.0


def test_cond_eager_false():
    pred = Tensor(np.array(False), TensorConfig((), DType.Bool, Device("cpu")))

    def true_fn():
        return Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))

    def false_fn():
        return Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))

    res = cond_eager(pred, true_fn, false_fn)
    assert res.numpy() == 0.0


def test_while_loop_eager():
    def cond_fn(x):
        return Tensor(x[0].numpy() < 5, TensorConfig((), DType.Bool, Device("cpu")))

    def body_fn(x):
        return [Tensor(x[0].numpy() + 1, TensorConfig((), DType.Float32, Device("cpu")))]

    init_val = Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = while_loop_eager(cond_fn, body_fn, [init_val])
    assert res[0].numpy() == 5.0


def test_scan_eager():
    from ml_switcheroo_compiler.ops.control_flow.eager import scan_eager

    def scan_fn(carry, x):
        new_carry = Tensor(carry[0].numpy() + x.numpy(), TensorConfig((), DType.Float32, Device("cpu")))
        return [new_carry], [new_carry]

    init = [Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))]
    xs = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, Device("cpu")))

    res_carry, res_ys = scan_eager(scan_fn, init, xs)
    assert res_carry[0].numpy() == 6.0


def test_map_fn_eager():
    from ml_switcheroo_compiler.ops.control_flow.eager import map_fn_eager

    def fn(x):
        return Tensor(x.numpy() * 2, TensorConfig((), DType.Float32, Device("cpu")))

    elems = Tensor(np.array([1.0, 2.0, 3.0]), TensorConfig((3,), DType.Float32, Device("cpu")))

    res = map_fn_eager(fn, elems)
    np.testing.assert_array_equal(res.numpy(), np.array([2.0, 4.0, 6.0]))


def test_pmap_eager():
    from ml_switcheroo_compiler.ops.control_flow.eager import pmap_eager

    @pmap_eager
    def fn(x):
        return Tensor(x.numpy() * 2, TensorConfig((), DType.Float32, Device("cpu")))

    x = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, Device("cpu")))
    res = fn(x)
    np.testing.assert_array_equal(res.numpy(), np.array([2.0, 4.0]))


def test_stop_gradient_eager():
    from ml_switcheroo_compiler.ops.control_flow.eager import stop_gradient_eager

    x = Tensor(np.array(1.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = stop_gradient_eager(x)
    assert res.config.requires_grad is False
    assert res.numpy() == 1.0


def test_assert_value_eager():
    from ml_switcheroo_compiler.ops.control_flow.eager import assert_value_eager

    x = Tensor(np.array(True), TensorConfig((), DType.Bool, Device("cpu")))
    assert_value_eager(x, "test msg")


def test_while_loop_eager_non_tuple():
    def cond_fn(x):
        return Tensor(x.numpy() < 5, TensorConfig((), DType.Bool, Device("cpu")))

    def body_fn(x):
        return Tensor(x.numpy() + 1, TensorConfig((), DType.Float32, Device("cpu")))

    init_val = Tensor(np.array(0.0), TensorConfig((), DType.Float32, Device("cpu")))
    res = while_loop_eager(cond_fn, body_fn, init_val)
    assert res.numpy() == 5.0


from ml_switcheroo_compiler.ops.control_flow.eager import map_fn_eager, scan_eager


def test_scan_tuple_return():
    # test scan returning a tuple for y
    def f(carry, x):
        return carry, (x, x)

    init = Tensor(np.array(0), TensorConfig((), DType.Int32, None))
    xs = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, None))

    with patch("ml_switcheroo_compiler.ops.control_flow.eager.get_active_backend") as mock_backend:
        backend = MagicMock()
        backend.execute_op.return_value = MagicMock(shape=(2, 2))
        mock_backend.return_value = backend

        carry, y = scan_eager(f, init, xs)
        assert y is not None


def test_map_fn_tuple_return():
    def f(x):
        return (x, x)

    xs = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, None))
    with patch("ml_switcheroo_compiler.ops.control_flow.eager.get_active_backend") as mock_backend:
        backend = MagicMock()
        backend.execute_op.return_value = MagicMock(shape=(2, 2))
        mock_backend.return_value = backend

        y = map_fn_eager(f, xs)
        assert y is not None
