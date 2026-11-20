"""Control flow tests."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import while_loop
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from tests.testing_utils import coerce_numeric, extract_test_data

if typing.TYPE_CHECKING:
    pass
device = Device(DeviceType.CPU, 0)


def test_while_loop_eager() -> None:
    """Test the while loop eager behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):

        def cond_fn(val: int) -> Tensor:
            """Cond fn.

            Args:
                val: Loop counter value.

            Returns:
                Tensor: Boolean scalar continuation flag.
            """
            return Tensor(val < 5, TensorConfig((), DType.Bool, device))

        def body_fn(val: int) -> int:
            """Body fn.

            Args:
                val: Loop counter value.

            Returns:
                int: Incremented counter value.
            """
            return val + 1

        res = while_loop(cond_fn, body_fn, 0)
        assert res == 5


def test_while_loop_trace() -> None:
    """Test the while loop trace behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):

        def cond_fn(val: Tensor) -> Tensor:
            """Cond fn.

            Args:
                val: Loop tensor state.

            Returns:
                Tensor: Symbolic boolean predicate tensor.
            """
            return Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device))

        def body_fn(val: Tensor) -> Tensor:
            """Body fn.

            Args:
                val: Loop tensor state.

            Returns:
                Tensor: Next tensor state.
            """
            return val

        init_val = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Float32, device))
        global_tracing_state.start_tracing()
        res = while_loop(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        global_tracing_state.stop_tracing()


def test_while_loop_tuple_init() -> None:
    """Test the while loop tuple init behavior.

    Returns:
        OpValue: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        t1 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))
        t2 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))

        def cond_fn(state: tuple[Tensor, Tensor]) -> Tensor:
            """Cond fn.

            Args:
                state: Tuple of state tensors.

            Returns:
                Tensor: Boolean continuation flag.
            """
            (v1, _v2) = state
            return Tensor(float(coerce_numeric(extract_test_data(v1))) < 2, TensorConfig((), DType.Bool, device))

        def body_fn(state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            """Body fn.

            Args:
                state: Tuple of state tensors.

            Returns:
                tuple[Tensor, Tensor]: Updated state tuple.
            """
            (v1, v2) = state
            return (
                Tensor(coerce_numeric(extract_test_data(v1)) + 1, TensorConfig((), DType.Int32, device)),
                v2,
            )

        res_pair = while_loop(cond_fn, body_fn, (t1, t2))
        assert isinstance(res_pair, tuple)
        res1, _res2 = res_pair
        assert res1.data == 2
    with ConfigContext(eager_mode=False):
        pt1 = Tensor(ProxyTensor(id="mock1", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))
        pt2 = Tensor(ProxyTensor(id="mock2", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))

        def cond_fn_trace(v1: Tensor, v2: Tensor) -> Tensor:
            """Cond fn trace.

            Args:
                v1: First state tensor.
                v2: Second state tensor.

            Returns:
                Tensor: Symbolic predicate tensor.
            """
            return Tensor(ProxyTensor(id="mock3", shape=(), dtype="bool"), TensorConfig((), DType.Bool, device))

        def body_fn_trace(v1: Tensor, v2: Tensor) -> tuple[Tensor, Tensor]:
            """Body fn trace.

            Args:
                v1: First state tensor.
                v2: Second state tensor.

            Returns:
                tuple[Tensor, Tensor]: Updated traced state tuple.
            """
            return (v1, v2)

        global_tracing_state.start_tracing()
        try:
            from ml_switcheroo_compiler.core.shape import OpValue
            from tests.testing_utils import require_callable

            res_trace: OpValue = require_callable(while_loop)(cond_fn_trace, body_fn_trace, [pt1, pt2])
            assert isinstance(res_trace, list)
            assert len(res_trace) == 2
        finally:
            global_tracing_state.stop_tracing()
