"""Tests for MLX parity."""

import numpy as np

import ml_switcheroo_compiler.grad as grad_module
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import (
    Device,
    DeviceType,
    FunctionExporter,
    Stream,
    StreamContext,
    clear_cache,
    export_function,
    exporter,
)
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig


def test_mlx_parity_coverage() -> None:
    """Tests the MLX specific API parity."""
    config.eager_mode = True
    t = Tensor(np.array(1), TensorConfig((), "float32", Device(DeviceType.CPU)))

    # ArrayAt
    assert t.at[0].add(1) is t
    assert t.at[0].multiply(1) is t
    assert t.at[0].set(1) is t
    assert t.at[0].maximum(1) is t
    assert t.at[0].minimum(1) is t

    # Stream, StreamContext
    s = Stream()
    with StreamContext(s):
        pass

    clear_cache()

    with FunctionExporter():
        pass
    export_function()
    assert isinstance(exporter(), FunctionExporter)

    # grad
    def f(x: object) -> object:
        """Docstring."""
        return x

    assert grad_module.jvp(f, [1], [1]) == (1, [1])
    assert grad_module.custom_vjp(f)(1) == 1

    # value_and_grad already tested in jax, but test again
    assert grad_module.value_and_grad(f)(1) == (1, 1)

    # lazy eval
    assert t.eval() is t


def test_device_dunders() -> None:
    """Tests Device dunder methods."""
    d1 = Device(DeviceType.CPU, 0)
    d2 = Device(DeviceType.CPU, 0)
    d3 = Device(DeviceType.GPU, 1)

    assert d1 == d2
    assert d1 != d3
    assert d1 != "not a device"
    assert hash(d1) == hash(d2)
    assert repr(d1) == "Device(cpu:0)"


def test_grad_coverage_extra() -> None:
    """Tests extra grad methods."""

    def f(x: object) -> object:
        """Docstring."""
        return x

    assert grad_module.grad(f)(1) == 1
    assert grad_module.jit(f)(1) == 1
    with grad_module.disable_jit():
        pass
    assert grad_module.eval_shape(f, 1) == 1
