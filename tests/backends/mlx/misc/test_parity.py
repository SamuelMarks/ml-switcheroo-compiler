"""Tests for MLX parity."""

from __future__ import annotations

import typing

import numpy as np

import ml_switcheroo_compiler.grad as grad_module
from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.device import Device, DeviceType, FunctionExporter, Stream, StreamContext, clear_cache, export_function, exporter
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_mlx_parity_coverage() -> None:
    """Test the mlx parity coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    t = Tensor(np.array(1), TensorConfig((), "float32", Device(DeviceType.CPU)))
    assert t.at[0].add(1) is t
    assert t.at[0].multiply(1) is t
    assert t.at[0].set(1) is t
    assert t.at[0].maximum(1) is t
    assert t.at[0].minimum(1) is t
    t1d = Tensor(np.array([2.0, 4.0]), TensorConfig((2,), "float32", Device(DeviceType.CPU)))
    assert typing.cast("EagerTensorProtocol", t1d.at[0].add(3.0).data).tolist() == [5.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].multiply(2.0).data).tolist() == [10.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].set(2.0).data).tolist() == [2.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].maximum(5.0).data).tolist() == [5.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].maximum(1.0).data).tolist() == [5.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].minimum(1.0).data).tolist() == [1.0, 4.0]
    assert typing.cast("EagerTensorProtocol", t1d.at[0].minimum(5.0).data).tolist() == [1.0, 4.0]
    s = Stream()
    with StreamContext(s):
        pass
    clear_cache()
    with FunctionExporter():
        pass
    export_function()
    assert isinstance(exporter(), FunctionExporter)

    def f(x: EagerValue) -> EagerValue:
        """Evaluate and process the f operation.

        Args:
            x (EagerValue): Required parameter for x.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return x

    assert grad_module.jvp(f, [1], [1]) == (1, [1])
    assert typing.cast("EagerCallableProtocol", grad_module.custom_vjp)(f)(1) == 1
    assert typing.cast("EagerCallableProtocol", grad_module.value_and_grad)(f)(1) == (1, 1)
    assert t.eval() is t


def test_device_dunders() -> None:
    """Test the device dunders behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    d1 = Device(DeviceType.CPU, 0)
    d2 = Device(DeviceType.CPU, 0)
    d3 = Device(DeviceType.GPU, 1)
    assert d1 == d2
    assert d1 != d3
    assert typing.cast("EagerTensorProtocol", d1) != typing.cast("EagerTensorProtocol", "not a device")
    assert hash(d1) == hash(d2)
    assert repr(d1) == "Device(cpu:0)"


def test_grad_coverage_extra() -> None:
    """Test the grad coverage extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    def f(x: EagerValue) -> EagerValue:
        """Evaluate and process the f operation.

        Args:
            x (EagerValue): Required parameter for x.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return x

    assert grad_module.grad(f)(1) == 1
    assert typing.cast("EagerCallableProtocol", grad_module.jit)(f)(1) == 1
    with grad_module.disable_jit():
        pass
    assert grad_module.eval_shape(f, 1) == 1
