"""Docstring module."""

from typing import Any
from ml_switcheroo.tracing import ProxyTensor
import numpy as np
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.device import Device, DeviceType
from ml_switcheroo.core.config import ConfigContext
from ml_switcheroo.control_flow import cond, while_loop, scan, vmap, pmap

device = Device(DeviceType.CPU)


def test_cond_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        pred = Tensor(np.array(True), (), DType.Bool, device)
        res = cond(pred, lambda: 1, lambda: 0)
        assert res == 1

        pred2 = Tensor(np.array(False), (), DType.Bool, device)
        res2 = cond(pred2, lambda: 1, lambda: 0)
        assert res2 == 0


def test_while_loop_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):

        def cond_fn(val: Any) -> Any:
            return Tensor(val < 5, (), DType.Bool, device)

        def body_fn(val: Any) -> Any:
            return val + 1

        res = while_loop(cond_fn, body_fn, 0)
        assert res == 5


def test_scan_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def f(carry: Any, x: Any) -> Any:
            y = carry + x.data
            return y, Tensor(y, (), DType.Int32, device)

        carry, ys = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ys.data, np.array([1, 3, 6]))


def test_vmap_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t: Any) -> Any:
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = vmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_pmap_eager() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t: Any) -> Any:
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = pmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_cond_trace() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        pred = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Bool, device
        )

        def true_fn() -> Any:
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Float32,
                device,
            )

        def false_fn() -> Any:
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


def test_while_loop_trace() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):

        def cond_fn(val: Any) -> Any:
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Bool,
                device,
            )

        def body_fn(val: Any) -> Any:
            return val

        init_val = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Float32, device
        )
        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = while_loop(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()


def test_scan_trace() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        xs = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )
        init = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Int32, device
        )

        def f(carry: Any, x: Any) -> Any:
            return carry, x

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        carry, ys = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        _tracer.stop_tracing()


def test_vmap_trace() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )

        def func(t: Any) -> Any:
            return t

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = vmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()


def test_pmap_trace() -> None:
    """Docstring."""
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (3,), DType.Int32, device
        )

        def func(t: Any) -> Any:
            return t

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        res = pmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()


def test_trace_function_type_error() -> None:
    """Docstring."""
    from ml_switcheroo.control_flow import _trace_function
    from ml_switcheroo.tracing import _tracer
    import pytest

    def bad_func(*args: Any) -> int:
        return 42

    try:
        with pytest.raises(
            TypeError,
            match="Control flow functions must return a Tensor or a tuple of Tensors.",
        ):  # noqa: E501
            _trace_function(bad_func, (), "test")
    finally:
        _tracer.is_tracing = False
        _tracer.active_graph = None


def test_control_flow_outside_tracing() -> None:
    """Docstring."""
    from ml_switcheroo.core.config import ConfigContext
    from ml_switcheroo.control_flow import cond, while_loop, scan, vmap, pmap
    import pytest

    with ConfigContext(eager_mode=False):
        pred = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), (), DType.Bool, device
        )  # noqa: E501

        with pytest.raises(
            RuntimeError, match="Cannot emit Cond node outside of a tracing context."
        ):  # noqa: E501
            cond(pred, lambda: None, lambda: None)

        with pytest.raises(
            RuntimeError, match="Cannot emit While node outside of a tracing context."
        ):  # noqa: E501
            while_loop(lambda x: pred, lambda x: x, pred)

        with pytest.raises(
            RuntimeError, match="Cannot emit Scan node outside of a tracing context."
        ):  # noqa: E501
            scan(lambda c, x: (c, x), pred, pred)

        with pytest.raises(
            RuntimeError, match="Cannot emit Vmap outside of a tracing context."
        ):  # noqa: E501
            vmap(lambda x: x)(pred)

        with pytest.raises(
            RuntimeError, match="Cannot emit Pmap outside of a tracing context."
        ):  # noqa: E501
            pmap(lambda x: x)(pred)


def test_while_loop_tuple_init() -> None:
    """Docstring."""
    from ml_switcheroo.core.config import ConfigContext
    from ml_switcheroo.control_flow import while_loop
    import numpy as np

    with ConfigContext(eager_mode=True):
        t1 = Tensor(np.array(0), (), DType.Int32, device)
        t2 = Tensor(np.array(0), (), DType.Int32, device)

        def cond_fn(state: Any) -> Any:
            v1, v2 = state
            return Tensor(v1.data < 2, (), DType.Bool, device)

        def body_fn(state: Any) -> Any:
            v1, v2 = state
            return (Tensor(v1.data + 1, (), DType.Int32, device), v2)

        res1, res2 = while_loop(cond_fn, body_fn, (t1, t2))
        assert res1.data == 2

    with ConfigContext(eager_mode=False):
        pt1 = Tensor(
            ProxyTensor(id="mock1", shape=(), dtype="int32"), (), DType.Int32, device
        )  # noqa: E501
        pt2 = Tensor(
            ProxyTensor(id="mock2", shape=(), dtype="int32"), (), DType.Int32, device
        )  # noqa: E501

        def cond_fn_trace(v1: Any, v2: Any) -> Any:
            return Tensor(
                ProxyTensor(id="mock3", shape=(), dtype="bool"), (), DType.Bool, device
            )  # noqa: E501

        def body_fn_trace(v1: Any, v2: Any) -> Any:
            return (v1, v2)

        from ml_switcheroo.tracing import _tracer

        _tracer.start_tracing()
        try:
            res_trace = while_loop(cond_fn_trace, body_fn_trace, [pt1, pt2])
            assert isinstance(res_trace, list)
            assert len(res_trace) == 2
        finally:
            _tracer.stop_tracing()
