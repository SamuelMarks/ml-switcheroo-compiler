"""Tests for test_main."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import pmap, vmap
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_vmap_in_axes_length_coverage() -> None:
    """Test the vmap in axes length coverage behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):

        def my_func(a: EagerValue, b: EagerValue) -> EagerValue:
            """Evaluate and process the my func operation.

            Args:
                a (EagerValue): Required parameter for a.
                b (EagerValue): Required parameter for b.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return as_tensor(a) + as_tensor(b)

        t1 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        t2 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        v_func = vmap(my_func, in_axes=(0,))
        with pytest.raises(RuntimeError):
            as_callable(v_func)(t1, t2)


def test_vmap_pmap_scalar_fallback() -> None:
    """Test the vmap pmap scalar fallback behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Evaluate and process the f operation.

        Args:
            x (EagerValue): Required parameter for x.
            y (EagerValue): Required parameter for y.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return as_tensor(x) * as_tensor(y)

    dev = Device("cpu")
    t = Tensor([1, 2, 3], TensorConfig((3,), DType.Int32, dev))
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        v_f = as_callable(vmap)(f, in_axes=(0, None))
        with pytest.raises(AttributeError):
            as_callable(v_f)(t, 5)
        p_f = pmap(f)
        as_callable(p_f)(t, 5)
        global_tracing_state.stop_tracing()
