# ruff: noqa: E501
import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device, DeviceType
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import TracingError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops.control_flow import AssertOp, assert_value, cond, map_fn, pmap, scan, stop_gradient, vmap, while_loop
from ml_switcheroo_compiler.ops.control_flow.tracing import _flatten_inputs, assert_value_tracing, map_fn_tracing, pmap_tracing
from ml_switcheroo_compiler.ops.control_flow_utils import _trace_function
from ml_switcheroo_compiler.ops.unary.arithmetic import Negative
from ml_switcheroo_compiler.tracing import ProxyTensor, global_tracing_state

"Extra tests for control flow."


def test_stop_gradient_eager_extra() -> object:
    """Test the stop gradient eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert stop_gradient(t) is t
    finally:
        config.eager_mode = False


def test_stop_gradient_tracing_extra() -> object:
    """Test the stop gradient tracing extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert stop_gradient(t) is t
        pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
        res = stop_gradient(pt)
        assert isinstance(res, ProxyTensor)
    finally:
        global_tracing_state.stop_tracing()


def test_pmap_eager_extra() -> object:
    """Test the pmap eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Float32, dev))

        def f(x: object) -> object:
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return x

        res = pmap(f)(t)
        assert isinstance(res, Tensor)
    finally:
        config.eager_mode = False


def test_assert_value_eager_extra() -> object:
    """Test the assert value eager extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert_value(t, "msg")
    finally:
        config.eager_mode = False


def test_assert_value_tracing_extra() -> object:
    """Test the assert value tracing extra behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        dev = Device(DeviceType.CPU)
        pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
        t = Tensor(pt, TensorConfig((1,), DType.Float32, dev))
        assert_value(t, "msg")
        assert_value(pt, "msg")
    finally:
        global_tracing_state.stop_tracing()


def test_scan_tuple_return() -> object:
    """Test the scan tuple return behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        xs = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(c: object, x: object) -> object:
            """Evaluate and process the f operation.

            Args:
                c (object): Required parameter for c.
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return (c, (x, x))

        scan(f, xs, xs)
    finally:
        config.eager_mode = False


def test_map_tuple_return() -> object:
    """Test the map tuple return behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        elems = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(x: object) -> object:
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return (x, x)

        map_fn(f, elems)
    finally:
        config.eager_mode = False


def test_tracing_exceptions() -> object:
    """Test the tracing exceptions behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
    config.eager_mode = False
    with pytest.raises((RuntimeError, TracingError, Exception)):
        map_fn_tracing(lambda x: x, t)
    with pytest.raises((RuntimeError, TracingError, Exception)):
        pmap_tracing(lambda x: x)(t)
    assert_value_tracing(t, "msg")
    pt = ProxyTensor(id="id", shape=(), dtype="float32")
    t2 = Tensor(pt, t.config)
    res = _flatten_inputs([t2, (t2,)])
    assert len(res) == 2
    res2 = _flatten_inputs(1)
    assert res2 == []


def test_assert_op_infer_shape() -> object:
    """Test the assert op infer shape behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    assert AssertOp().infer_shape(None) == ()


def test_pmap_tracing_non_tensor() -> object:
    """Test the pmap tracing non tensor behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        pt = ProxyTensor(id="id", shape=(1,), dtype="float32")
        t2 = Tensor(pt, t.config)
        pmap_tracing(lambda x, y: x)(t2, 42)
    finally:
        global_tracing_state.stop_tracing()


"Unit tests for control flow operations in eager and tracing modes.\n\nThis module contains tests for cond, while_loop, scan, vmap, and pmap operations,\nverifying their behavior in both eager execution mode and during tracer-based graph\nconstruction.\n"

device = Device(DeviceType.CPU)


def test_vmap_eager() -> None:
    """Test the vmap eager behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests the eager execution of the vectorized map (vmap) operator.\n\n    Verifies that vmap correctly applies a function element-wise or along a\n    batch dimension when eager mode is enabled\n\n    Returns:\n    None\n    "
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
    """Test the pmap eager behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests the eager execution of the parallel map (pmap) operator.\n\n    Verifies that pmap correctly maps a function over a tensor in eager mode\n\n    Returns:\n    None\n    "
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
    """Test the vmap trace behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests the tracing behavior of the vectorized map (vmap) operator.\n\n    Verifies that vmap correctly records the vectorization operation into the\n    active tracing graph when eager mode is disabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=False):
        x = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))

        def func(t: object) -> object:
            """Func.

            Args:
            t (object): The t parameter

            Returns:
            object: The resulting output.
            """
            return t

        global_tracing_state.start_tracing()
        res = vmap(func)(x)
        assert res.dtype == DType.Int32
        global_tracing_state.stop_tracing()


def test_pmap_trace() -> None:
    """Test the pmap trace behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests the tracing behavior of the parallel map (pmap) operator.\n\n    Verifies that pmap correctly records the parallel map operation into the\n    active tracing graph when eager mode is disabled\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=False):
        x = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))

        def func(t: object) -> object:
            """Func.

            Args:
            t (object): The t parameter

            Returns:
            object: The resulting output.
            """
            return t

        global_tracing_state.start_tracing()
        res = pmap(func)(x)
        assert res.dtype == DType.Int32
        global_tracing_state.stop_tracing()


def test_trace_function_type_error() -> None:
    """Test the trace function type error behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests that tracing a function that returns an invalid type raises a TypeError.\n\n    Verifies that _trace_function enforces that control flow functions must\n    return a Tensor or a tuple of Tensors, raising a TypeError otherwise\n\n    Returns:\n    None\n    "

    def bad_func(*args: object) -> int:
        """Bad func.

        Args:
        *args (object): Additional keyword arguments.

        Returns:
        int: The resulting output.
        """
        return 42

    try:
        with pytest.raises(TypeError, match="Control flow functions must return a Tensor or a tuple of Tensors."):
            _trace_function(bad_func, (), "test")
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None


def test_control_flow_outside_tracing() -> None:
    """Test the control flow outside tracing behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Tests that control flow operators raise errors when called outside a tracing.\n\n    context\n\n    Verifies that calling cond, while_loop, scan, vmap, or pmap in non-eager\n    mode without an active tracer raises a RuntimeError\n\n    Returns:\n    None\n    "
    with ConfigContext(eager_mode=False):
        pred = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device))
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Cond node outside of a tracing context."):
            cond(pred, lambda: pred, lambda: pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit While node outside of a tracing context."):
            while_loop(lambda x: pred, lambda x: x, pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Scan node outside of a tracing context."):
            scan(lambda c, x: (c, x), pred, pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Vmap outside of a tracing context."):
            vmap(lambda x: x)(pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Pmap outside of a tracing context."):
            pmap(lambda x: x)(pred)


def test_stop_gradient() -> None:
    """Test the stop gradient behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Test stop_gradient."
    device = Device(DeviceType.CPU)
    t_eager = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType.Float32, device))
    t_out = stop_gradient(t_eager)
    assert t_out is t_eager
    graph = global_tracing_state.start_tracing(name="stop_gradient_test")
    try:
        proxy = ProxyTensor(id="input_proxy", shape=(2,), dtype="float32")
        out_proxy = stop_gradient(proxy)
        assert isinstance(out_proxy, ProxyTensor)
        assert out_proxy.id != proxy.id
        node = graph.nodes[out_proxy.id]
        assert node.op_type == "StopGradient"
        assert node.inputs == ["input_proxy"]
        t_trace = Tensor(proxy, TensorConfig((2,), DType.Float32, device))
        out_trace = stop_gradient(t_trace)
        assert isinstance(out_trace, Tensor)
        assert isinstance(out_trace.data, ProxyTensor)
        assert out_trace.data.id != proxy.id
        node2 = graph.nodes[out_trace.data.id]
        assert node2.op_type == "StopGradient"
        assert node2.inputs == ["input_proxy"]
        val = 42.0
        assert stop_gradient(val) is val
    finally:
        global_tracing_state.stop_tracing()


def test_vmap_tuple_axes() -> None:
    """Test the vmap tuple axes behavior.

    Returns:
        Any: The inferred shape or computed result.
    """
    "Test vmap with tuple in_axes."
    device = Device("cpu")
    with ConfigContext(eager_mode=True):

        def f(x: object) -> object:
            """Evaluate and process the f operation.

            Args:
                x (object): Required parameter for x.

            Returns:
                object: The evaluated or processed output.
            """
            return Negative()(x)

        vmap_f = vmap(f, in_axes=(0,), out_axes=(0,))
        x = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))
        y = vmap_f(x)
        assert np.array_equal(y.data, np.array([-1, -2]))


def test_new_control_flow_fori() -> object:
    """Test the new control flow fori behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True

    def body_fun(i: object, x: object) -> object:
        """Evaluate and process the body fun operation.

        Args:
            i (object): Required parameter for i.
            x (object): Required parameter for x.

        Returns:
            object: The evaluated or processed output.
        """
        return ops.add(x, 1.0)

    init_val = ops.array(np.array(0.0).astype(np.float32))
    lower = ops.array(np.array(0).astype(np.int32))
    upper = ops.array(np.array(5).astype(np.int32))
    res = ops.fori_loop(lower, upper, body_fun, init_val)
    assert res is not None


def test_new_control_flow_map() -> object:
    """Test the new control flow map behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    elems = ops.array(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    res_map = ops.map(lambda x: ops.multiply(x, 2.0), elems)
    res_vmap = ops.vectorized_map(lambda x: ops.multiply(x, 2.0), elems)
    assert res_map is not None
    assert res_vmap is not None


def test_new_control_flow_switch() -> object:
    """Test the new control flow switch behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    index = ops.array(np.array(1).astype(np.int32))
    branches = [lambda x: ops.multiply(x, 1.0), lambda x: ops.multiply(x, 2.0), lambda x: ops.multiply(x, 3.0)]
    arg = ops.array(np.array(10.0).astype(np.float32))
    res_switch = ops.switch(index, branches, arg)
    assert res_switch is not None


def test_new_control_flow_custom_gradient() -> object:
    """Test the new control flow custom gradient behavior.

    Returns:
        object: The inferred shape or computed result.
    """
    config.eager_mode = True
    arg = ops.array(np.array(10.0).astype(np.float32))

    @ops.custom_gradient
    def my_fn(x: object) -> object:
        """Evaluate and process the my fn operation.

        Args:
            x (object): Required parameter for x.

        Returns:
            object: The evaluated or processed output.
        """
        return (ops.multiply(x, 2.0), lambda g: ops.multiply(g, 2.0))

    res_custom = my_fn(arg)
    assert res_custom is not None
