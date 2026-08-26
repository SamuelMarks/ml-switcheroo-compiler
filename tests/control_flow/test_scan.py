# ruff: noqa: E501
"""Control flow tests."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import scan
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

device = Device(DeviceType.CPU, 0)


def test_scan_eager() -> None:
    """Test the scan eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    "Tests the eager execution of the scan operator.\n\n    Verifies that scan correctly loops over a tensor, carrying state and\n    accumulating results, when eager mode is enabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry, x):
            """F.

            Args:
            carry (object): The carry parameter
            x (object): The first input tensor.

            Returns:
            object: The resulting output.
            """
            y = carry + x.data
            return (y, Tensor(y, TensorConfig((), DType.Int32, device)))

        (carry, ys) = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ys.data, np.array([1, 3, 6]))


def test_scan_trace() -> None:
    """Test the scan trace behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    "Tests the tracing behavior of the scan operator.\n\n    Verifies that scan correctly records the scan operation into the active\n    tracing graph when eager mode is disabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=False):
        xs = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))
        init = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Int32, device))

        def f(carry, x):
            """F.

            Args:
            carry (object): The carry parameter
            x (object): The first input tensor.

            Returns:
            object: The resulting output.
            """
            return (carry, x)

        global_tracing_state.start_tracing()
        (_carry, ys) = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        global_tracing_state.stop_tracing()


def test_scan_eager_ndarray() -> None:
    """Test the scan eager ndarray behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    "Test scan eager mode returning an ndarray."
    device = Device("cpu")
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry, x):
            """Evaluate and process the f operation.

            Args:
                carry (object): Required parameter for carry.
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return (carry, Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device)))

        (carry_out, ys_out) = scan(f, 0, xs)
        assert isinstance(ys_out, Tensor)
        assert np.array_equal(ys_out.data, np.array([[1, 2], [1, 2], [1, 2]]))
