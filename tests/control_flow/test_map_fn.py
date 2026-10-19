"""Control flow map_fn tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import map_fn
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue
device = Device(DeviceType.CPU, 0)


def test_map_fn_eager() -> None:
    """Test the map fn eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            y = getattr(x, "data") * 2
            return Tensor(y, TensorConfig((), DType.Int32, device))

        ys = map_fn(f, xs)
        assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerTensorProtocol", getattr(ys, "data")), cast("EagerCallableProtocol", np.array)([2, 4, 6]))


def test_map_fn_trace() -> None:
    """Test the map fn trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        xs = Tensor(ProxyTensor(id="mock", shape=(), dtype="int32"), TensorConfig((3,), DType.Int32, device))

        def f(x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return x

        global_tracing_state.start_tracing()
        ys = map_fn(f, xs)
        assert cast("EagerTensorProtocol", getattr(ys, "dtype")) == DType.Int32
        global_tracing_state.stop_tracing()
