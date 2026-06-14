"""Control flow tests."""

from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.ops.control_flow import cond
import numpy as np
from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.device import Device, DeviceType

device = Device(DeviceType.CPU, 0)
"""Control flow tests."""


def test_cond_eager() -> None:
    """Tests the eager execution of the conditional operator.

    Verifies that cond correctly evaluates the predicate and executes the
    corresponding branch function (true or false) when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        pred = Tensor(np.array(True), (), DType.Bool, device)
        res = cond(pred, lambda: 1, lambda: 0)
        assert res == 1

        pred2 = Tensor(np.array(False), (), DType.Bool, device)
        res2 = cond(pred2, lambda: 1, lambda: 0)
        assert res2 == 0


def test_cond_trace() -> None:
    """Tests the tracing behavior of the conditional operator.

    Verifies that cond correctly records the conditional operation into the
    active tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):
        pred = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            (),
            DType.Bool,
            device,
        )

        def true_fn() -> object:
            """True fn.

            Returns:
                object: The resulting output.
            """
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Float32,
                device,
            )

        def false_fn() -> object:
            """False fn.

            Returns:
                object: The resulting output.
            """
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                (),
                DType.Float32,
                device,
            )

        from ml_switcheroo_compiler.tracing.tracer import _tracer

        _tracer.start_tracing()
        res = cond(pred, true_fn, false_fn)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()
