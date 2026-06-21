"""Control flow tests."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import scan
from ml_switcheroo_compiler.tracing import ProxyTensor

device = Device(DeviceType.CPU, 0)
"""Control flow tests."""


def test_scan_eager() -> None:
    """Tests the eager execution of the scan operator.

    Verifies that scan correctly loops over a tensor, carrying state and
    accumulating results, when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: object, x: object) -> object:
            """F.

            Args:
                carry (object): The carry parameter
                x (object): The first input tensor.

            Returns:
                object: The resulting output.
            """
            y = carry + x.data
            return y, Tensor(y, TensorConfig((), DType.Int32, device))

        carry, ys = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ys.data, np.array([1, 3, 6]))


def test_scan_trace() -> None:
    """Tests the tracing behavior of the scan operator.

    Verifies that scan correctly records the scan operation into the active
    tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):
        xs = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            TensorConfig((3,), DType.Int32, device),
        )
        init = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Int32, device)
        )

        def f(carry: object, x: object) -> object:
            """F.

            Args:
                carry (object): The carry parameter
                x (object): The first input tensor.

            Returns:
                object: The resulting output.
            """
            return carry, x

        from ml_switcheroo_compiler.tracing.tracer import _tracer

        _tracer.start_tracing()
        _carry, ys = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        _tracer.stop_tracing()


def test_scan_eager_ndarray() -> None:
    """Test scan eager mode returning an ndarray."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.control_flow import scan

    device = Device("cpu")
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(carry: object, x: object) -> object:
            """Docstring."""
            # Return a Tensor
            return carry, Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))

        carry_out, ys_out = scan(f, 0, xs)
        assert isinstance(ys_out, Tensor)
        assert np.array_equal(ys_out.data, np.array([[1, 2], [1, 2], [1, 2]]))
