# ruff: noqa: E501
"""Control flow map_fn tests."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import map_fn
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

device = Device(DeviceType.CPU, 0)


def test_map_fn_eager() -> None:
    """Test the map fn eager behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    "Tests the eager execution of the map_fn operator.\n\n    Verifies that map_fn correctly loops over a tensor, applying a function\n    and stacking results, when eager mode is enabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(x):
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            y = x.data * 2
            return Tensor(y, TensorConfig((), DType.Int32, device))

        ys = map_fn(f, xs)
        assert np.array_equal(ys.data, np.array([2, 4, 6]))


def test_map_fn_trace() -> None:
    """Test the map fn trace behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    "Tests the tracing behavior of the map_fn operator.\n\n    Verifies that map_fn correctly records the map operation into the active\n    tracing graph when eager mode is disabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=False):
        xs = Tensor(ProxyTensor(id="mock", shape=(), dtype="int32"), TensorConfig((3,), DType.Int32, device))

        def f(x):
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return x

        global_tracing_state.start_tracing()
        ys = map_fn(f, xs)
        assert ys.dtype == DType.Int32
        global_tracing_state.stop_tracing()
