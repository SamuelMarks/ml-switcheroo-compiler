"""Control flow map_fn tests."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import map_fn
from ml_switcheroo_compiler.tracing import ProxyTensor

device = Device(DeviceType.CPU, 0)


def test_map_fn_eager() -> None:
    """Tests the eager execution of the map_fn operator.

    Verifies that map_fn correctly loops over a tensor, applying a function
    and stacking results, when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def f(x: object) -> object:
            y = x.data * 2
            return Tensor(y, TensorConfig((), DType.Int32, device))

        ys = map_fn(f, xs)
        assert np.array_equal(ys.data, np.array([2, 4, 6]))


def test_map_fn_trace() -> None:
    """Tests the tracing behavior of the map_fn operator.

    Verifies that map_fn correctly records the map operation into the active
    tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):
        xs = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="int32"),
            TensorConfig((3,), DType.Int32, device),
        )

        def f(x: object) -> object:
            return x

        from ml_switcheroo_compiler.tracing.tracer import _tracer

        _tracer.start_tracing()
        ys = map_fn(f, xs)
        assert ys.dtype == DType.Int32
        _tracer.stop_tracing()
