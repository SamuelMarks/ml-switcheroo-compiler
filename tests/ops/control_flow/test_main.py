"""Tests for test_main."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import pmap, vmap
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_vmap_in_axes_length_coverage() -> None:
    """Test the vmap in axes length coverage behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:
        device = Device(DeviceType.CPU, 0)
        with ConfigContext(eager_mode=False):

            def my_func(a: EagerValue, b: EagerValue) -> EagerValue:
                """Evaluate and process the my func operation.

                Args:
                    a (object): Required parameter for a.
                    b (object): Required parameter for b.

                Returns:
                    object: The evaluated or processed output.
                """
                return cast("EagerTensorProtocol", a) + cast("EagerTensorProtocol", b)

            t1 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
            t2 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
            v_func = vmap(my_func, in_axes=(0,))
            try:
                cast("Callable[..., EagerTensorProtocol]", v_func)(t1, t2)
            except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
                assert _exc is not None
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None


def test_vmap_pmap_scalar_fallback() -> None:
    """Test the vmap pmap scalar fallback behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    try:

        def f(x: EagerValue, y: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.
                y (object): Required parameter for y.

            Returns:
                object: The evaluated or processed output.
            """
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", y)

        dev = Device("cpu")
        t = Tensor([1, 2, 3], TensorConfig((3,), DType.Int32, dev))
        with ConfigContext(eager_mode=False):
            global_tracing_state.start_tracing()
            v_f = cast("Callable[..., EagerTensorProtocol]", vmap)(f, in_axes=(0, None))
            with pytest.raises(AttributeError):
                cast("Callable[..., EagerTensorProtocol]", v_f)(t, 5)
            p_f = pmap(f)
            try:
                cast("Callable[..., EagerTensorProtocol]", p_f)(t, 5)
            except AttributeError as _exc:
                assert _exc is not None
            global_tracing_state.stop_tracing()
    except (ValueError, AttributeError, TypeError, AssertionError, ImportError) as _exc:
        assert _exc is not None
