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
from tests.testing_utils import coerce_numeric, require_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_vmap_in_axes_length_coverage() -> None:
    """Test the vmap in axes length coverage behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    device = Device(DeviceType.CPU, 0)
    with ConfigContext(eager_mode=False):

        def my_func(a: BackendArg, b: BackendArg) -> BackendArg:
            """Evaluate and process the my func operation.

            Args:
                a (BackendArg): Required parameter for a.
                b (BackendArg): Required parameter for b.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return coerce_numeric(a) + coerce_numeric(b)

        t1 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        t2 = Tensor(np.zeros((2, 3)), TensorConfig((2, 3), DType.Float32, device))
        v_func = vmap(my_func, in_axes=(0,))
        with pytest.raises(RuntimeError):
            require_callable(v_func)(t1, t2)


def test_vmap_pmap_scalar_fallback() -> None:
    """Test the vmap pmap scalar fallback behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """

    def f(x: Tensor, y: Tensor | int) -> Tensor:
        """Evaluate and process the f operation.

        Args:
            x (Tensor): Required parameter for x.
            y (Tensor | int): Required parameter for y.

        Returns:
            Tensor: The evaluated or processed output.
        """
        return x * y

    dev = Device("cpu")
    t = Tensor([1, 2, 3], TensorConfig((3,), DType.Int32, dev))
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing()
        v_f = require_callable(vmap)(f, in_axes=(0, None))
        with pytest.raises(AttributeError):
            require_callable(v_f)(t, 5)
        p_f = pmap(f)
        require_callable(p_f)(t, 5)
        global_tracing_state.stop_tracing()
