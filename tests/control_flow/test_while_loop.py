"""Control flow tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import while_loop
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue
device = Device(DeviceType.CPU, 0)


def test_while_loop_eager() -> None:
    """Test the while loop eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):

        def cond_fn(val: EagerValue) -> EagerValue:
            """Cond fn.

            Args:
            val (EagerValue): The val parameter

            Returns:
            EagerValue: The resulting output.
            """
            return cast("Callable[..., EagerTensorProtocol]", Tensor)(cast("EagerTensorProtocol", val) < cast("EagerTensorProtocol", 5), cast("Callable[..., EagerTensorProtocol]", TensorConfig)((), DType.Bool, device))

        def body_fn(val: EagerValue) -> EagerValue:
            """Body fn.

            Args:
            val (EagerValue): The val parameter

            Returns:
            EagerValue: The resulting output.
            """
            return cast("EagerTensorProtocol", val) + cast("EagerTensorProtocol", 1)

        res = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn, body_fn, 0)
        assert res == 5


def test_while_loop_trace() -> None:
    """Test the while loop trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):

        def cond_fn(val: EagerValue) -> EagerValue:
            """Cond fn.

            Args:
            val (EagerValue): The val parameter

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device))

        def body_fn(val: EagerValue) -> EagerValue:
            """Body fn.

            Args:
            val (EagerValue): The val parameter

            Returns:
            EagerValue: The resulting output.
            """
            return val

        init_val = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Float32, device))
        global_tracing_state.start_tracing()
        res = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        global_tracing_state.stop_tracing()


def test_while_loop_tuple_init() -> None:
    """Test the while loop tuple init behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        t1 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))
        t2 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))

        def cond_fn(state: EagerValue) -> EagerValue:
            """Cond fn.

            Args:
            state (EagerValue): The state parameter

            Returns:
            EagerValue: The resulting output.
            """
            (v1, _v2) = cast("EagerTensorProtocol", state)
            return Tensor(cast("EagerTensorProtocol", getattr(v1, "data")) < 2, TensorConfig((), DType.Bool, device))

        def body_fn(state: EagerValue) -> EagerValue:
            """Body fn.

            Args:
            state (EagerValue): The state parameter

            Returns:
            EagerValue: The resulting output.
            """
            (v1, v2) = cast("EagerTensorProtocol", state)
            return (Tensor(cast("EagerTensorProtocol", getattr(v1, "data")) + 1, TensorConfig((), DType.Int32, device)), v2)

        (res1, _res2) = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn, body_fn, (t1, t2))
        assert res1.data == 2
    with ConfigContext(eager_mode=False):
        pt1 = Tensor(ProxyTensor(id="mock1", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))
        pt2 = Tensor(ProxyTensor(id="mock2", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))

        def cond_fn_trace(v1: EagerValue, v2: EagerValue) -> EagerValue:
            """Cond fn trace.

            Args:
            v1 (EagerValue): The v1 parameter
            v2 (EagerValue): The v2 parameter

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(ProxyTensor(id="mock3", shape=(), dtype="bool"), TensorConfig((), DType.Bool, device))

        def body_fn_trace(v1: EagerValue, v2: EagerValue) -> EagerValue:
            """Body fn trace.

            Args:
            v1 (EagerValue): The v1 parameter
            v2 (EagerValue): The v2 parameter

            Returns:
            EagerValue: The resulting output.
            """
            return (v1, v2)

        global_tracing_state.start_tracing()
        try:
            res_trace = cast("Callable[..., EagerTensorProtocol]", while_loop)(cond_fn_trace, body_fn_trace, [pt1, pt2])
            assert isinstance(res_trace, list)
            assert len(res_trace) == 2
        finally:
            global_tracing_state.stop_tracing()
