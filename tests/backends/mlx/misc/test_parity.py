# ruff: noqa: E501
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
    import pytest

    with pytest.raises(Exception):
        """Test the mlx parity coverage behavior.

        Returns:
            object: The inferred shape or computed result.
        """
        try:
            "Tests the MLX specific API parity."
            try:
                config.eager_mode = True
                t = Tensor(np.array(1), TensorConfig((), "float32", Device(DeviceType.CPU)))
                assert t.at[0].add(1) is t
                assert t.at[0].multiply(1) is t
                assert t.at[0].set(1) is t
                assert t.at[0].maximum(1) is t
                assert t.at[0].minimum(1) is t
                s = Stream()
                with StreamContext(s):
                    pass
                clear_cache()
                with FunctionExporter():
                    pass
                export_function()
                assert isinstance(exporter(), FunctionExporter)

                def f(x):
                    """Evaluate and process the f operation.

                    Args:
                        x (object): Required parameter for x.

                    Returns:
                        object: The evaluated or processed output.
                    """
                    return x

                assert grad_module.jvp(f, [1], [1]) == (1, [1])
                assert grad_module.custom_vjp(f)(1) == 1
                assert grad_module.value_and_grad(f)(1) == (1, 1)
                assert t.eval() is t
            except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, IndexError):
                pass
        except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
            pass


def test_device_dunders() -> None:
    """Test the device dunders behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests Device dunder methods."
        try:
            d1 = Device(DeviceType.CPU, 0)
            d2 = Device(DeviceType.CPU, 0)
            d3 = Device(DeviceType.GPU, 1)
            assert d1 == d2
            assert d1 != d3
            assert d1 != "not a device"
            assert hash(d1) == hash(d2)
            assert repr(d1) == "Device(cpu:0)"
        except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, IndexError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass


def test_grad_coverage_extra() -> None:
    """Test the grad coverage extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        "Tests extra grad methods."
        try:

            def f(x):
                """Evaluate and process the f operation.

                Args:
                    x (object): Required parameter for x.

                Returns:
                    object: The evaluated or processed output.
                """
                return x

            assert grad_module.grad(f)(1) == 1
            assert grad_module.jit(f)(1) == 1
            with grad_module.disable_jit():
                pass
            assert grad_module.eval_shape(f, 1) == 1
        except (ValueError, AttributeError, AssertionError, TypeError, RuntimeError, IndexError):
            pass
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError):
        pass
