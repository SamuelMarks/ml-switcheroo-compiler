from ml_switcheroo.tracing import ProxyTensor
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device, DeviceType
from ml_switcheroo.core.config import ConfigContext
from ml_switcheroo.control_flow import cond, while_loop, scan, vmap, pmap

device = Device(DeviceType.CPU)


def test_cond_eager():
    with ConfigContext(eager_mode=True):
        pred = Tensor(np.array(True), (), DType.Bool, device)
        res = cond(pred, lambda: 1, lambda: 0)
        assert res == 1

        pred2 = Tensor(np.array(False), (), DType.Bool, device)
        res2 = cond(pred2, lambda: 1, lambda: 0)
        assert res2 == 0


def test_while_loop_eager():
    with ConfigContext(eager_mode=True):

        def cond_fn(val):
            return Tensor(val < 5, (), DType.Bool, device)

        def body_fn(val):
            return val + 1

        res = while_loop(cond_fn, body_fn, 0)
        assert res == 5


def test_scan_eager():
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def f(carry, x):
            y = carry + x.data
            return y, Tensor(y, (), DType.Int32, device)

        carry, ys = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ys.data, np.array([1, 3, 6]))


def test_vmap_eager():
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t):
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = vmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_pmap_eager():
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t):
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = pmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_cond_trace():
    with ConfigContext(eager_mode=False):
        pred = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Bool, device
        )

        def true_fn():
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Float32,
                device,
            )

        def false_fn():
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Float32,
                device,
            )

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = cond(pred, true_fn, false_fn)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()


def test_while_loop_trace():
    with ConfigContext(eager_mode=False):

        def cond_fn(val):
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Bool,
                device,
            )

        def body_fn(val):
            return val

        init_val = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Float32, device
        )
        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = while_loop(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()


def test_scan_trace():
    with ConfigContext(eager_mode=False):
        xs = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )
        init = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Int32, device
        )

        def f(carry, x):
            return carry, x

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        carry, ys = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        _tracer.stop_tracing()


def test_vmap_trace():
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )

        def func(t):
            return t

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = vmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()


def test_pmap_trace():
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )

        def func(t):
            return t

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = pmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()
