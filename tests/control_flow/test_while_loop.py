"""Control flow tests."""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import while_loop
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state

device = Device(DeviceType.CPU, 0)


def test_while_loop_eager() -> None:
    """Tests the eager execution of the while_loop operator.

    Verifies that while_loop correctly iterates using the condition and body
    functions until the condition is no longer met when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):

        def cond_fn(val: object) -> object:
            """Cond fn.

            Args:
                val (object): The val parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(val < 5, TensorConfig((), DType.Bool, device))

        def body_fn(val: object) -> object:
            """Body fn.

            Args:
                val (object): The val parameter

            Returns:
                object: The resulting output.
            """
            return val + 1

        res = while_loop(cond_fn, body_fn, 0)
        assert res == 5


def test_while_loop_trace() -> None:
    """Tests the tracing behavior of the while_loop operator.

    Verifies that while_loop correctly records the loop operation into the
    active tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):

        def cond_fn(val: object) -> object:
            """Cond fn.

            Args:
                val (object): The val parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(
                ProxyTensor(id="mock", shape=(), dtype="float32"),
                TensorConfig((), DType.Bool, device),
            )

        def body_fn(val: object) -> object:
            """Body fn.

            Args:
                val (object): The val parameter

            Returns:
                object: The resulting output.
            """
            return val

        init_val = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            TensorConfig((), DType.Float32, device),
        )

        global_tracing_state.start_tracing()
        res = while_loop(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        global_tracing_state.stop_tracing()


def test_while_loop_tuple_init() -> None:
    """Tests the while_loop operator with tuple and list initial states.

    Verifies that while_loop correctly handles structured state (tuples in
    eager mode, lists in tracing mode) for both condition and body functions

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        t1 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))
        t2 = Tensor(np.array(0), TensorConfig((), DType.Int32, device))

        def cond_fn(state: object) -> object:
            """Cond fn.

            Args:
                state (object): The state parameter

            Returns:
                object: The resulting output.
            """
            v1, _v2 = state
            return Tensor(v1.data < 2, TensorConfig((), DType.Bool, device))

        def body_fn(state: object) -> object:
            """Body fn.

            Args:
                state (object): The state parameter

            Returns:
                object: The resulting output.
            """
            v1, v2 = state
            return (Tensor(v1.data + 1, TensorConfig((), DType.Int32, device)), v2)

        res1, _res2 = while_loop(cond_fn, body_fn, (t1, t2))
        assert res1.data == 2

    with ConfigContext(eager_mode=False):
        pt1 = Tensor(ProxyTensor(id="mock1", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))
        pt2 = Tensor(ProxyTensor(id="mock2", shape=(), dtype="int32"), TensorConfig((), DType.Int32, device))

        def cond_fn_trace(v1: object, v2: object) -> object:
            """Cond fn trace.

            Args:
                v1 (object): The v1 parameter
                v2 (object): The v2 parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(
                ProxyTensor(id="mock3", shape=(), dtype="bool"),
                TensorConfig((), DType.Bool, device),
            )

        def body_fn_trace(v1: object, v2: object) -> object:
            """Body fn trace.

            Args:
                v1 (object): The v1 parameter
                v2 (object): The v2 parameter

            Returns:
                object: The resulting output.
            """
            return (v1, v2)

        global_tracing_state.start_tracing()
        try:
            res_trace = while_loop(cond_fn_trace, body_fn_trace, [pt1, pt2])
            assert isinstance(res_trace, list)
            assert len(res_trace) == 2
        finally:
            global_tracing_state.stop_tracing()
