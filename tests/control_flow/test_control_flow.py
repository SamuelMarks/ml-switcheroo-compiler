"""Tests for test_control_flow."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler import ops
from ml_switcheroo_compiler.backends.eager_registry import EagerValue
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
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_stop_gradient_eager_extra() -> None:
    """Test the stop gradient eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert stop_gradient(t) is t
    finally:
        config.eager_mode = False


def test_stop_gradient_tracing_extra() -> None:
    """Test the stop gradient tracing extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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


def test_pmap_eager_extra() -> None:
    """Test the pmap eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([[1, 2], [3, 4]]), TensorConfig((2, 2), DType.Float32, dev))

        def f(x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return x

        res = as_callable(pmap)(f)(t)
        assert isinstance(as_tensor(res), Tensor)
    finally:
        config.eager_mode = False


def test_assert_value_eager_extra() -> None:
    """Test the assert value eager extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
        assert_value(t, "msg")
    finally:
        config.eager_mode = False


def test_assert_value_tracing_extra() -> None:
    """Test the assert value tracing extra behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
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


def test_scan_tuple_return() -> None:
    """Test the scan tuple return behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        xs = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(c: EagerValue, x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                c (EagerValue): Required parameter for c.
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return (c, (x, x))

        as_callable(scan)(f, xs, xs)
    finally:
        config.eager_mode = False
    assert callable(test_scan_tuple_return)


def test_map_tuple_return() -> None:
    """Test the map tuple return behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    try:
        dev = Device(DeviceType.CPU)
        elems = Tensor(np.array([[1], [2]]), TensorConfig((2, 1), DType.Float32, dev))

        def f(x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return (x, x)

        map_fn(f, elems)
    finally:
        config.eager_mode = False
    assert callable(test_map_tuple_return)


def test_tracing_exceptions() -> None:
    """Test the tracing exceptions behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
    config.eager_mode = False
    with pytest.raises((RuntimeError, TracingError, Exception)):
        map_fn_tracing(lambda x: x, t)
    with pytest.raises((RuntimeError, TracingError, Exception)):
        as_callable(pmap_tracing)(lambda x: x)(t)
    assert_value_tracing(t, "msg")
    pt = ProxyTensor(id="id", shape=(), dtype="float32")
    t2 = Tensor(pt, t.config)
    res = _flatten_inputs([t2, (t2,)])
    assert len(res) == 2
    res2 = _flatten_inputs(1)
    assert res2 == []


def test_assert_op_infer_shape() -> None:
    """Test the assert op infer shape behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    assert as_callable(AssertOp)().infer_shape(None) == ()


def test_pmap_tracing_non_tensor() -> None:
    """Test the pmap tracing non tensor behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    dev = Device(DeviceType.CPU)
    t = Tensor(np.array([1]), TensorConfig((1,), DType.Float32, dev))
    config.eager_mode = False
    global_tracing_state.start_tracing()
    try:
        pt = ProxyTensor(id="id", shape=(1,), dtype="float32")
        t2 = Tensor(pt, t.config)
        as_callable(pmap_tracing)(lambda x, y: x)(t2, 42)
    finally:
        global_tracing_state.stop_tracing()
    assert t.shape == (1,)


device = Device(DeviceType.CPU)


def test_vmap_eager() -> None:
    """Test the vmap eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def func(t: EagerValue) -> EagerValue:
            """Func.

            Args:
            t (EagerValue): The t parameter

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(getattr(t, "data") * 2, TensorConfig(getattr(t, "shape"), getattr(t, "dtype"), getattr(t, "device")))

        res = as_callable(vmap)(func)(x)
        assert as_callable(np.array_equal)(as_tensor(getattr(res, "data")), as_callable(np.array)([2, 4, 6]))


def test_pmap_eager() -> None:
    """Test the pmap eager behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=True):
        x = Tensor(np.array([1, 2, 3]), TensorConfig((3,), DType.Int32, device))

        def func(t: EagerValue) -> EagerValue:
            """Func.

            Args:
            t (EagerValue): The t parameter

            Returns:
            EagerValue: The resulting output.
            """
            return Tensor(getattr(t, "data") * 2, TensorConfig(getattr(t, "shape"), getattr(t, "dtype"), getattr(t, "device")))

        res = as_callable(pmap)(func)(x)
        assert as_callable(np.array_equal)(as_tensor(getattr(res, "data")), as_callable(np.array)([2, 4, 6]))


def test_vmap_trace() -> None:
    """Test the vmap trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        x = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))

        def func(t: EagerValue) -> EagerValue:
            """Func.

            Args:
            t (EagerValue): The t parameter

            Returns:
            EagerValue: The resulting output.
            """
            return t

        global_tracing_state.start_tracing()
        res = as_callable(vmap)(func)(x)
        assert as_tensor(getattr(res, "dtype")) == DType.Int32
        global_tracing_state.stop_tracing()


def test_pmap_trace() -> None:
    """Test the pmap trace behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        x = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((3,), DType.Int32, device))

        def func(t: EagerValue) -> EagerValue:
            """Func.

            Args:
            t (EagerValue): The t parameter

            Returns:
            EagerValue: The resulting output.
            """
            return t

        global_tracing_state.start_tracing()
        res = as_callable(pmap)(func)(x)
        assert as_tensor(getattr(res, "dtype")) == DType.Int32
        global_tracing_state.stop_tracing()


def test_trace_function_type_error() -> None:
    """Test the trace function type error behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """

    def bad_func(*args: EagerValue) -> int:
        """Bad func.

        Args:
        *args (EagerValue): Additional keyword arguments.

        Returns:
        int: The resulting output.
        """
        return 42

    try:
        with pytest.raises(TypeError, match="Control flow functions must return a Tensor or a tuple of Tensors."):
            as_callable(_trace_function)(bad_func, (), "test")
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None


def test_control_flow_outside_tracing() -> None:
    """Test the control flow outside tracing behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    with ConfigContext(eager_mode=False):
        pred = Tensor(ProxyTensor(id="mock", shape=(), dtype="float32"), TensorConfig((), DType.Bool, device))
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Cond node outside of a tracing context."):
            cond(pred, lambda: pred, lambda: pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit While node outside of a tracing context."):
            while_loop(lambda x: pred, lambda x: x, pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Scan node outside of a tracing context."):
            scan(lambda c, x: (c, x), pred, pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Vmap outside of a tracing context."):
            as_callable(vmap)(lambda x: x)(pred)
        with pytest.raises((RuntimeError, TracingError, Exception), match="Cannot emit Pmap outside of a tracing context."):
            as_callable(pmap)(lambda x: x)(pred)


def test_stop_gradient() -> None:
    """Test the stop gradient behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
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
        assert id(as_callable(stop_gradient)(val)) == id(val)
    finally:
        global_tracing_state.stop_tracing()


def test_vmap_tuple_axes() -> None:
    """Test the vmap tuple axes behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    device = Device("cpu")
    with ConfigContext(eager_mode=True):

        def f(x: EagerValue) -> EagerValue:
            """Evaluate and process the f operation.

            Args:
                x (EagerValue): Required parameter for x.

            Returns:
                EagerValue: The evaluated or processed output.
            """
            return Negative()(x)

        vmap_f = vmap(f, in_axes=(0,), out_axes=(0,))
        x = Tensor(np.array([1, 2]), TensorConfig((2,), DType.Int32, device))
        y = as_callable(vmap_f)(x)
        assert as_callable(np.array_equal)(as_tensor(getattr(y, "data")), as_callable(np.array)([-1, -2]))


def test_new_control_flow_fori() -> None:
    """Test the new control flow fori behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True

    def body_fun(i: EagerValue, x: EagerValue) -> EagerValue:
        """Evaluate and process the body fun operation.

        Args:
            i (EagerValue): Required parameter for i.
            x (EagerValue): Required parameter for x.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return ops.add(x, 1.0)

    init_val = ops.array(np.array(0.0).astype(np.float32))
    lower = ops.array(np.array(0).astype(np.int32))
    upper = ops.array(np.array(5).astype(np.int32))
    res = ops.fori_loop(lower, upper, body_fun, init_val)
    assert res is not None


def test_new_control_flow_map() -> None:
    """Test the new control flow map behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    elems = ops.array(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    res_map = ops.map(lambda x: ops.multiply(x, 2.0), elems)
    res_vmap = ops.vectorized_map(lambda x: ops.multiply(x, 2.0), elems)
    assert res_map is not None
    assert res_vmap is not None


def test_new_control_flow_switch() -> None:
    """Test the new control flow switch behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    index = ops.array(np.array(1).astype(np.int32))
    branches = [lambda x: ops.multiply(x, 1.0), lambda x: ops.multiply(x, 2.0), lambda x: ops.multiply(x, 3.0)]
    arg = ops.array(np.array(10.0).astype(np.float32))
    res_switch = ops.switch(index, branches, arg)
    assert res_switch is not None


def test_new_control_flow_custom_gradient() -> None:
    """Test the new control flow custom gradient behavior.

    Returns:
        TensorOrScalar: The inferred shape or computed result.
    """
    config.eager_mode = True
    arg = ops.array(np.array(10.0).astype(np.float32))

    @ops.custom_gradient
    def my_fn(x: EagerValue) -> EagerValue:
        """Evaluate and process the my fn operation.

        Args:
            x (EagerValue): Required parameter for x.

        Returns:
            EagerValue: The evaluated or processed output.
        """
        return (ops.multiply(x, 2.0), lambda g: ops.multiply(g, 2.0))

    res_custom = as_callable(my_fn)(arg)
    assert res_custom is not None


def test_switch_empty() -> None:
    """Test switch_empty."""
    import pytest

    from ml_switcheroo_compiler.ops.control_flow import switch

    with pytest.raises(ValueError, match="branches cannot be empty"):
        as_callable(switch)(0, [])


def test_map_fn_lazy() -> None:
    """Test map_fn_lazy."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import map_fn

    orig = config.eager_mode
    try:
        config.eager_mode = False
        import numpy as np

        from ml_switcheroo_compiler.core.device import Device, DeviceType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t = Tensor(np.array([1.0]), TensorConfig((1,), "float32", Device(DeviceType.CPU)))
        with pytest.raises((RuntimeError, ValueError, TypeError, KeyError, Exception)):
            map_fn(lambda x: x, t)
    finally:
        config.eager_mode = orig


def test_dyn_custom_grad_infer_shape_and_vjp() -> None:
    """Test dyn_custom_grad_infer_shape_and_vjp."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import custom_gradient

    @custom_gradient
    def my_op(x: EagerValue) -> EagerValue:
        """Test function my_op.

        Args:
            x: Test parameter `x`.
        """

        class Val:
            """Test class Val."""

            shape = (1,)
            dtype = "float32"

        return (Val(), lambda g: g)

    orig = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state

        graph = global_tracing_state.start_tracing()
        try:
            pt = ProxyTensor(id="pt", shape=(1,), dtype="float32")
            out = as_callable(my_op)(pt)
            assert out is not None
            node = as_tensor(getattr(graph.nodes, "__getitem__")(as_tensor(getattr(out, "_data")).id))
            op_obj = getattr(node, "op", getattr(node, "op_type", None))
            from ml_switcheroo_compiler.ops.registry import get_op

            OpCls = as_callable(get_op)(op_obj)
            inst = OpCls()
            assert inst.infer_shape() == (1,)
            from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import get_vjp

            vjp_fn = as_callable(get_vjp)(op_obj)
            assert vjp_fn(graph, None, "cot") == tuple()
        finally:
            global_tracing_state.stop_tracing()
    finally:
        config.eager_mode = orig


def test_case_and_switch_case() -> None:
    """Test case_and_switch_case."""
    import pytest

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import case, switch_case

    orig = config.eager_mode
    try:
        config.eager_mode = True
        with pytest.raises(ValueError, match="case requires at least one"):
            case([])
        assert as_callable(case)([], default=lambda: 42) == 42
        with pytest.raises(ValueError, match="switch_case requires at least one branch"):
            from ml_switcheroo_compiler.core.device import Device, DeviceType
            from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

            t = Tensor(0, TensorConfig((), "int32", Device(DeviceType.CPU)))
            switch_case(t, {})
        from ml_switcheroo_compiler.core.device import Device, DeviceType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t = Tensor(0, TensorConfig((), "int32", Device(DeviceType.CPU)))
        assert switch_case(t, {}, default=lambda: 42) == 42
        import numpy as np

        t_index = Tensor(np.array(1), TensorConfig((), "int32", Device(DeviceType.CPU)))
        as_callable(switch_case)(t_index, {0: lambda: 10, 1: lambda: 20})
    finally:
        config.eager_mode = orig


def test_infer_shapes() -> None:
    """Test infer_shapes."""
    from ml_switcheroo_compiler.ops.control_flow import AssociativeScan, DebugInfs, DebugNans, ScanOp, SwitchOp

    assert DebugInfs().infer_shape() == ()
    assert DebugInfs().infer_shape(1) == 1
    assert DebugNans().infer_shape() == ()
    assert DebugNans().infer_shape(2) == 2
    assert as_callable(SwitchOp)().infer_shape(None, None) == ()
    assert ScanOp().infer_shape() == ()
    assert AssociativeScan().infer_shape() == ()
    assert AssociativeScan().infer_shape(3) == 3


def test_associative_scan_eager() -> None:
    """Test associative_scan_eager."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import associative_scan

    orig = config.eager_mode
    try:
        config.eager_mode = True
        import numpy as np

        from ml_switcheroo_compiler.core.device import Device, DeviceType
        from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

        t = Tensor(np.array([1.0]), TensorConfig((1,), "float32", Device(DeviceType.CPU)))
        associative_scan(t)
    finally:
        config.eager_mode = orig
    assert callable(test_associative_scan_eager)


def test_associative_scan_lazy() -> None:
    """Test associative_scan_lazy."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.ops.control_flow import associative_scan

    orig = config.eager_mode
    try:
        config.eager_mode = False
        from ml_switcheroo_compiler.tracing.tracer import ProxyTensor, global_tracing_state

        global_tracing_state.start_tracing()
        try:
            pt = ProxyTensor(id="pt", shape=(2, 2), dtype="float32")
            out = associative_scan(pt)
            assert getattr(out, "shape", as_tensor(getattr(out, "_data")).shape) == (2, 2)
        finally:
            global_tracing_state.stop_tracing()
    finally:
        config.eager_mode = orig


def test_scan_bind() -> None:
    """Test scan_bind."""
    from ml_switcheroo_compiler.ops.control_flow import scan_bind

    (f, xs) = as_callable(scan_bind)(lambda x: x, [1, 2], 3, kw=4)
    assert xs == [1, 2]
