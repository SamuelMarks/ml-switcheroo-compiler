"""Unit tests for control flow operations in eager and tracing modes.

This module contains tests for cond, while_loop, scan, vmap, and pmap operations,
verifying their behavior in both eager execution mode and during tracer-based graph
construction.
"""

import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
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
        x = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(t.data * 2, TensorConfig(t.shape, t.dtype, t.device))

        res = vmap(func)(x)
        assert np.array_equal(res.data, np.array([2, 4, 6]))


def test_pmap_eager() -> None:
    """Tests the eager execution of the parallel map (pmap) operator.

    Verifies that pmap correctly maps a function over a tensor in eager mode

    Returns:
    None
    """
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def func(t: object) -> object:
            """Func.

            Args:
                t (object): The t parameter

            Returns:
                object: The resulting output.
            """
            return Tensor(t.data * 2, TensorConfig(t.shape, t.dtype, t.device))

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
            TensorConfig((3,), DType.Int32, device),
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
            TensorConfig((3,), DType.Int32, device),
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

    from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
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
            ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device)
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
    t_eager = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
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
        t_trace = Tensor(proxy, TensorConfig((2,), DType.Float32, device))
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
        x = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))
        y = vmap_f(x)
        assert np.array_equal(y.data, np.array([-1, -2]))


def test_new_control_flow():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler import ops

    config.eager_mode = True

    # test fori_loop
    def body_fun(i, x):
        return ops.add(x, 1.0)

    init_val = ops.array(np.array(0.0).astype(np.float32))
    lower = ops.array(np.array(0).astype(np.int32))
    upper = ops.array(np.array(5).astype(np.int32))

    res = ops.fori_loop(lower, upper, body_fun, init_val)
    assert res is not None

    # test map and vectorized_map
    elems = ops.array(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    res_map = ops.map(lambda x: ops.multiply(x, 2.0), elems)
    res_vmap = ops.vectorized_map(lambda x: ops.multiply(x, 2.0), elems)
    assert res_map is not None
    assert res_vmap is not None

    # test switch
    index = ops.array(np.array(1).astype(np.int32))
    branches = [
        lambda x: ops.multiply(x, 1.0),
        lambda x: ops.multiply(x, 2.0),
        lambda x: ops.multiply(x, 3.0),
    ]
    arg = ops.array(np.array(10.0).astype(np.float32))
    res_switch = ops.switch(index, branches, arg)
    assert res_switch is not None

    # test custom_gradient
    @ops.custom_gradient
    def my_fn(x):
        return ops.multiply(x, 2.0), lambda g: ops.multiply(g, 2.0)

    res_custom = my_fn(arg)
    assert res_custom is not None
