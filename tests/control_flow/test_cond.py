"""Control flow tests."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import cond
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
device = Device(DeviceType.CPU, 0)


def test_cond_eager() -> None:
    """Test the cond eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        pred = Tensor(np.array(True), TensorConfig((), DType.Bool, device))
        res = cond(pred, lambda: 1, lambda: 0)
        assert res == 1
        pred2 = Tensor(np.array(False), TensorConfig((), DType.Bool, device))
        res2 = cond(pred2, lambda: 1, lambda: 0)
        assert res2 == 0


def test_cond_trace() -> None:
    """Test the cond trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        pred = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device))

        def true_fn() -> EagerValue:
            """True fn.

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Float32, device))

        def false_fn() -> EagerValue:
            """False fn.

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Float32, device))

        global_tracing_state.start_tracing()
        res = cond(pred, true_fn, false_fn)
        assert typing.cast("EagerTensorProtocol", getattr(res, "dtype")) == DType.Float32
        global_tracing_state.stop_tracing()
