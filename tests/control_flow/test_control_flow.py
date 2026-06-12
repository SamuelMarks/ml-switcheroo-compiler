"""Unit tests for control flow operations in eager and tracing modes.

This module contains tests for cond, while_loop, scan, vmap, and pmap operations,
verifying their behavior in both eager execution mode and during tracer-based graph
construction.
"""

import numpy as np

from ml_switcheroo.core.config import ConfigContext
from ml_switcheroo.core.device import Device, DeviceType
from ml_switcheroo.core.dtype import DType
from ml_switcheroo.core.tensor import Tensor
from ml_switcheroo.ops.control_flow import cond, pmap, scan, vmap, while_loop
from ml_switcheroo.tracing.tracer import ProxyTensor

device = Device(DeviceType.CPU)


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
            return Tensor(val < 5, (), DType.Bool, device)

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


def test_scan_eager() -> None:
    """Tests the eager execution of the scan operator.

    Verifies that scan correctly loops over a tensor, carrying state and
    accumulating results, when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        xs = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def f(carry: object, x: object) -> object:
            """F.

            Args:
                carry (object): The carry parameter
                x (object): The x parameter

            Returns:
                object: The resulting output.
            """
            y = carry + x.data
            return y, Tensor(y, (), DType.Int32, device)

        carry, ys = scan(f, 0, xs)
        assert carry == 6
        assert np.array_equal(ys.data, np.array([1, 3, 6]))


def test_vmap_eager() -> None:
    """Tests the eager execution of the vectorized map (vmap) operator.

    Verifies that vmap correctly applies a function element-wise or along a
    batch dimension when eager mode is enabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = vmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_pmap_eager() -> None:
    """Tests the eager execution of the parallel map (pmap) operator.

    Verifies that pmap correctly maps a function over a tensor in eager mode

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), (3,), DType.Int32, device)

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(t.data * 2, t.shape, t.dtype, t.device)

        res = pmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


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

        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        res = cond(pred, true_fn, false_fn)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()


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
                (),
                DType.Bool,
                device,
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
            (),
            DType.Float32,
            device,
        )
        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        res = while_loop(cond_fn, body_fn, init_val)
        assert res.dtype == DType.Float32
        _tracer.stop_tracing()


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
            (3,),
            DType.Int32,
            device,
        )
        init = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            (),
            DType.Int32,
            device,
        )

        def f(carry: object, x: object) -> object:
            """F.

            Args:
                carry (object): The carry parameter
                x (object): The x parameter

            Returns:
                object: The resulting output.
            """
            return carry, x

        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        _carry, ys = scan(f, init, xs)
        assert ys.dtype == DType.Int32
        _tracer.stop_tracing()


def test_vmap_trace() -> None:
    """Tests the tracing behavior of the vectorized map (vmap) operator.

    Verifies that vmap correctly records the vectorization operation into the
    active tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            (3,),
            DType.Int32,
            device,
        )

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return t

        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        res = vmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()


def test_pmap_trace() -> None:
    """Tests the tracing behavior of the parallel map (pmap) operator.

    Verifies that pmap correctly records the parallel map operation into the
    active tracing graph when eager mode is disabled

    Returns:
    None
    """
    with ConfigContext(eager_mode=False):
        x = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            (3,),
            DType.Int32,
            device,
        )

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return t

        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        res = pmap(func)(x)
        assert res.dtype == DType.Int32
        _tracer.stop_tracing()


def test_trace_function_type_error() -> None:
    """Tests that tracing a function that returns an invalid type raises a TypeError.

    Verifies that _trace_function enforces that control flow functions must
    return a Tensor or a tuple of Tensors, raising a TypeError otherwise

    Returns:
    None
    """
    import pytest

    from ml_switcheroo.ops.control_flow import _trace_function
    from ml_switcheroo.tracing.tracer import _tracer

    def bad_func(*args: object) -> int:
        """Bad func.

        Args:
            *args (object): Variable length argument list

        Returns:
            int: The resulting output.
        """
        return 42

    try:
        with pytest.raises(
            TypeError,
            match="Control flow functions must return a Tensor or a tuple of Tensors.",
        ):
            _trace_function(bad_func, (), "test")
    finally:
        _tracer.is_tracing = False
        _tracer.active_graph = None


def test_control_flow_outside_tracing() -> None:
    """Tests that control flow operators raise errors when called outside a tracing.

    context

    Verifies that calling cond, while_loop, scan, vmap, or pmap in non-eager
    mode without an active tracer raises a RuntimeError

    Returns:
    None
    """
    import pytest

    from ml_switcheroo.core.config import ConfigContext
    from ml_switcheroo.ops.control_flow import cond, pmap, scan, vmap, while_loop

    with ConfigContext(eager_mode=False):
        pred = Tensor(
            ProxyTensor(id="mock", shape=(), dtype="float32"),
            (),
            DType.Bool,
            device,
        )

        with pytest.raises(
            RuntimeError,
            match="Cannot emit Cond node outside of a tracing context.",
        ):
            cond(pred, lambda: None, lambda: None)

        with pytest.raises(
            RuntimeError,
            match="Cannot emit While node outside of a tracing context.",
        ):
            while_loop(lambda x: pred, lambda x: x, pred)

        with pytest.raises(
            RuntimeError,
            match="Cannot emit Scan node outside of a tracing context.",
        ):
            scan(lambda c, x: (c, x), pred, pred)

        with pytest.raises(
            RuntimeError,
            match="Cannot emit Vmap outside of a tracing context.",
        ):
            vmap(lambda x: x)(pred)

        with pytest.raises(
            RuntimeError,
            match="Cannot emit Pmap outside of a tracing context.",
        ):
            pmap(lambda x: x)(pred)


def test_while_loop_tuple_init() -> None:
    """Tests the while_loop operator with tuple and list initial states.

    Verifies that while_loop correctly handles structured state (tuples in
    eager mode, lists in tracing mode) for both condition and body functions

    Returns:
    None
    """
    import numpy as np

    from ml_switcheroo.core.config import ConfigContext
    from ml_switcheroo.ops.control_flow import while_loop

    with ConfigContext(eager_mode=True):
        t1 = Tensor(np.array(0), (), DType.Int32, device)
        t2 = Tensor(np.array(0), (), DType.Int32, device)

        def cond_fn(state: object) -> object:
            """Cond fn.

            Args:
                state (object): The state parameter

            Returns:
                object: The resulting output.
            """
            v1, _v2 = state
            return Tensor(v1.data < 2, (), DType.Bool, device)

        def body_fn(state: object) -> object:
            """Body fn.

            Args:
                state (object): The state parameter

            Returns:
                object: The resulting output.
            """
            v1, v2 = state
            return (Tensor(v1.data + 1, (), DType.Int32, device), v2)

        res1, _res2 = while_loop(cond_fn, body_fn, (t1, t2))
        assert res1.data == 2

    with ConfigContext(eager_mode=False):
        pt1 = Tensor(
            ProxyTensor(id="mock1", shape=(), dtype="int32"),
            (),
            DType.Int32,
            device,
        )
        pt2 = Tensor(
            ProxyTensor(id="mock2", shape=(), dtype="int32"),
            (),
            DType.Int32,
            device,
        )

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
                (),
                DType.Bool,
                device,
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

        from ml_switcheroo.tracing.tracer import _tracer

        _tracer.start_tracing()
        try:
            res_trace = while_loop(cond_fn_trace, body_fn_trace, [pt1, pt2])
            assert isinstance(res_trace, list)
            assert len(res_trace) == 2
        finally:
            _tracer.stop_tracing()
