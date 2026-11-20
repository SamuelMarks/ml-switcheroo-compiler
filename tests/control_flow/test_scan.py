"""Control flow tests."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import scan
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import coerce_numeric, extract_test_data

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
device = Device(DeviceType.CPU, 0)


def test_scan_eager() -> None:
    """Test the scan eager behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: BackendArg, x: BackendArg) -> BackendArg:
            """F.

            Args:
            carry (BackendArg): The carry parameter
            x (BackendArg): The first input tensor.

            Returns:
            BackendArg: The resulting output.
            """
            y = coerce_numeric(carry) + coerce_numeric(extract_test_data(x))
            return (y, Tensor(y, TensorConfig((), DType.Int32, device)))

        (carry, ys) = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ensure_ndarray(ys.data), np.array([1, 3, 6]))


def test_scan_trace() -> None:
    """Test the scan trace behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        xs = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))
        init = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Int32, device))

        def f(carry: BackendArg, x: BackendArg) -> BackendArg:
            """F.

            Args:
            carry (BackendArg): The carry parameter
            x (BackendArg): The first input tensor.

            Returns:
            BackendArg: The resulting output.
            """
            return (carry, x)

        global_tracing_state.start_tracing()
        (_carry, ys) = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        global_tracing_state.stop_tracing()


def test_scan_eager_ndarray() -> None:
    """Test the scan eager ndarray behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    device = Device("cpu")
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: BackendArg, x: BackendArg) -> BackendArg:
            """Evaluate and process the f operation.

            Args:
                carry (BackendArg): Required parameter for carry.
                x (BackendArg): Required parameter for x.

            Returns:
                BackendArg: The evaluated or processed output.
            """
            return (carry, Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device)))

        (carry_out, ys_out) = scan(f, 0, xs)
        assert isinstance(ys_out, Tensor)
        assert np.array_equal(ensure_ndarray(ys_out.data), np.array([[1, 2], [1, 2], [1, 2]]))
