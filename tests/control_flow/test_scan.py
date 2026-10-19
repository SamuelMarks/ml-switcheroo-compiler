"""Control flow tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import scan
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerValue
device = Device(DeviceType.CPU, 0)


def test_scan_eager() -> None:
    """Test the scan eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: EagerValue, x: EagerValue) -> EagerValue:
            """F.

            Args:
            carry (EagerValue): The carry parameter
            x (EagerValue): The first input tensor.

            Returns:
            EagerValue: The resulting output.
            """
            y = carry + getattr(x, "data")
            return (y, Tensor(y, TensorConfig((), DType.Int32, device)))

        (carry, ys) = cast("EagerCallableProtocol", scan)(f, 0, xs)
        assert carry == 6
        assert cast("EagerCallableProtocol", np.array_equal)(ys.data, cast("EagerCallableProtocol", np.array)([1, 3, 6]))


def test_scan_trace() -> None:
    """Test the scan trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        xs = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))
        init = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Int32, device))

        def f(carry: EagerValue, x: EagerValue) -> EagerValue:
            """F.

            Args:
            carry (EagerValue): The carry parameter
            x (EagerValue): The first input tensor.

            Returns:
            EagerValue: The resulting output.
            """
            return (carry, x)

        global_tracing_state.start_tracing()
        (_carry, ys) = cast("EagerCallableProtocol", scan)(f, init, xs)
        assert ys.dtype == DType.Int32
        global_tracing_state.stop_tracing()


def test_scan_eager_ndarray() -> None:
    """Test the scan eager ndarray behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device("cpu")
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: EagerValue, x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                carry (EagerValue): Required parameter for carry.
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return (carry, Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device)))

        (carry_out, ys_out) = cast("EagerCallableProtocol", scan)(f, 0, xs)
        assert isinstance(ys_out, Tensor)
        assert cast("EagerCallableProtocol", np.array_equal)(ys_out.data, cast("EagerCallableProtocol", np.array)([[1, 2], [1, 2], [1, 2]]))
