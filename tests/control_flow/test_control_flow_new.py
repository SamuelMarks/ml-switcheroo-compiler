"""Extra tests for control flow."""

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import AssertOp, assert_value, map_fn, pmap, scan, stop_gradient
from ml_switcheroo_compiler.ops.control_flow.tracing import (
    _flatten_inputs,
    assert_value_tracing,
    map_fn_tracing,
    pmap_tracing,
)
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state


def test_stop_gradient_eager_extra() -> object:
    """Function docstring."""
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert stop_gradient(t) is t
    finally:
        config.eager_mode = False


def test_stop_gradient_tracing_extra() -> object:
    """Function docstring."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        # Not a proxy tensor
        assert stop_gradient(t) is t

        # Just a ProxyTensor
        pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
        res = stop_gradient(pt)
        assert isinstance(res, ProxyTensor)
    finally:
        global_tracing_state.stop_tracing()


def test_pmap_eager_extra() -> object:
    """Function docstring."""
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Float32, dev))

        def f(x: object) -> object:
            """Function docstring."""
            return x

        res = pmap(f)(t)
        assert isinstance(res, Tensor)
    finally:
        config.eager_mode = False


def test_assert_value_eager_extra() -> object:
    """Function docstring."""
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert_value(t, "msg")
    finally:
        config.eager_mode = False


def test_assert_value_tracing_extra() -> object:
    """Function docstring."""
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        dev = Device(DeviceType.CPU)
        pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
        t = Tensor(pt, TensorConfig((1,), DType.Float32, dev))
        assert_value(t, "msg")

        # just proxy
        assert_value(pt, "msg")
    finally:
        global_tracing_state.stop_tracing()


def test_scan_tuple_return() -> object:
    """Function docstring."""
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        xs = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(c: object, x: object) -> object:
            """Function docstring."""
            return c, (x, x)

        scan(f, xs, xs)
    finally:
        config.eager_mode = False


def test_map_tuple_return() -> object:
    """Function docstring."""
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        elems = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(x: object) -> object:
            """Function docstring."""
            return (x, x)

        map_fn(f, elems)
    finally:
        config.eager_mode = False


def test_tracing_exceptions() -> object:
    """Function docstring."""
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))

    config.eager_mode = False

    with pytest.raises(RuntimeError):
        map_fn_tracing(lambda x: x, t)

    with pytest.raises(RuntimeError):
        pmap_tracing(lambda x: x)(t)

    # assert_value_tracing early return
    assert_value_tracing(t, "msg")

    # test _flatten_inputs with list/tuple and other types
    pt = ProxyTensor(id="id", shape=(), dtype="float32")
    t2 = Tensor(pt, t.config)
    res = _flatten_inputs([t2, (t2,)])
    assert len(res) == 2
    res2 = _flatten_inputs(1)
    assert res2 == []


def test_assert_op_infer_shape() -> object:
    """Function docstring."""
    assert AssertOp().infer_shape(None) == ()


def test_pmap_tracing_non_tensor() -> object:
    """Function docstring."""
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))

    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        pt = ProxyTensor(id="id", shape=(1,), dtype="float32")
        t2 = Tensor(pt, t.config)
        pmap_tracing(lambda x, y: x)(t2, 42)
    finally:
        global_tracing_state.stop_tracing()
