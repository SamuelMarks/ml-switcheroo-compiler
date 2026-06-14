"""Unit tests for control flow operations in eager and tracing modes.

This module contains tests for cond, while_loop, scan, vmap, and pmap operations,
verifying their behavior in both eager execution mode and during tracer-based graph
construction.
"""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor
from ml_switcheroo_compiler.ops.control_flow import cond, pmap, scan, vmap, while_loop
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor

device = Device(DeviceType.CPU)


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

        from ml_switcheroo_compiler.tracing.tracer import _tracer

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

        from ml_switcheroo_compiler.tracing.tracer import _tracer

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

    from ml_switcheroo_compiler.ops.control_flow import _trace_function
    from ml_switcheroo_compiler.tracing.tracer import _tracer

    def bad_func(*args: object) -> int:
        """Bad func.

        Args:
            *args (object): Additional keyword arguments.

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

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.control_flow import pmap, vmap

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


def test_stop_gradient() -> None:
    """Test stop_gradient."""
    import numpy as np

    from ml_switcheroo_compiler.core.device import Device, DeviceType
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.control_flow import stop_gradient
    from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, _tracer

    # Test eager mode
    device = Device(DeviceType.CPU)
    t_eager = Tensor(np.array([1.0, 2.0]), shape=(2,), dtype=DType.Float32, device=device)
    t_out = stop_gradient(t_eager)
    assert t_out is t_eager

    # Test tracing mode with ProxyTensor
    graph = _tracer.start_tracing(name="stop_gradient_test")
    try:
        proxy = ProxyTensor(id="input_proxy", shape=(2,), dtype="float32")
        out_proxy = stop_gradient(proxy)
        assert isinstance(out_proxy, ProxyTensor)
        assert out_proxy.id != proxy.id
        node = graph.nodes[out_proxy.id]
        assert node.op_type == "StopGradient"
        assert node.inputs == ["input_proxy"]

        # Test tracing mode with Tensor wrapping ProxyTensor
        t_trace = Tensor(proxy, shape=(2,), dtype=DType.Float32, device=device)
        out_trace = stop_gradient(t_trace)
        assert isinstance(out_trace, Tensor)
        assert isinstance(out_trace.data, ProxyTensor)
        assert out_trace.data.id != proxy.id
        node2 = graph.nodes[out_trace.data.id]
        assert node2.op_type == "StopGradient"
        assert node2.inputs == ["input_proxy"]

        # Test tracing mode with non-tensor
        val = 42.0
        assert stop_gradient(val) is val

    finally:
        _tracer.stop_tracing()


def test_vmap_tuple_axes() -> None:
    """Test vmap with tuple in_axes."""
    import numpy as np

    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ops.control_flow import vmap

    device = Device("cpu")
    with ConfigContext(eager_mode=True):

        def f(x: object) -> object:
            """Docstring."""
            from ml_switcheroo_compiler.ops.unary.math import Negative

            return Negative()(x)

        vmap_f = vmap(f, in_axes=(0,), out_axes=(0,))
        x = Tensor(np.array([1, 2]), (2,), DType.Int32, device)
        y = vmap_f(x)
        assert np.array_equal(y.data, np.array([-1, -2]))
