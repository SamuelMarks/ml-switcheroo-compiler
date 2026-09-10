"""Unit tests for custom JVP implementation and forward-mode autodiff integration."""

import numpy as np

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import CustomJVPFunction, custom_jvp, grad, jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_jvp_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import (
    has_jvp,
    load_primitive_jvp_rules,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    has_vjp,
    load_primitive_vjp_rules,
)


def test_custom_jvp_decorator_and_eager():
    """Verify custom_jvp decorator and eager evaluation with custom rule."""

    @custom_jvp
    def square_scaled(x):
        return ops.multiply(x, x)

    @square_scaled.defjvp
    def square_scaled_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        # Custom derivative: instead of 2 * x * t, use 42 * x * t
        primal_out = square_scaled(x)
        tan_out = ops.multiply(ops.multiply(x, 42.0), t)
        return primal_out, tan_out

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = jvp(square_scaled, (x,), (t,))

    val_arr = get_active_backend().asarray(val)
    tan_arr = get_active_backend().asarray(tan)

    np.testing.assert_allclose(val_arr, np.array([4.0], dtype=np.float32))
    np.testing.assert_allclose(tan_arr, np.array([84.0], dtype=np.float32))


def test_custom_jvp_graph_integration():
    """Verify custom JVP is properly handled within an outer traced graph."""

    @custom_jvp
    def cube_custom(x):
        return ops.multiply(x, ops.multiply(x, x))

    @cube_custom.defjvp
    def cube_custom_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        # Override derivative to 10.0 * t
        return cube_custom(x), ops.multiply(t, 10.0)

    def outer(x):
        return ops.add(cube_custom(x), 5.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = jvp(outer, (x,), (t,))

    tan_arr = get_active_backend().asarray(tan)
    # Derivative of outer w.r.t x should be 10.0 (from custom JVP rule)
    np.testing.assert_allclose(tan_arr, np.array([10.0], dtype=np.float32))


def test_custom_jvp_reverse_mode_grad():
    """Verify reverse-mode grad correctly propagates through CustomJVP nodes."""

    @custom_jvp
    def f_custom(x):
        return ops.multiply(x, x)

    @f_custom.defjvp
    def f_custom_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        return f_custom(x), ops.multiply(t, 2.0)

    def outer(x):
        return f_custom(x)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    res_grad = grad(outer)(x)

    np.testing.assert_allclose(get_active_backend().asarray(res_grad), np.array([6.0], dtype=np.float32))


def test_custom_jvp_has_aux():
    """Verify custom JVP with has_aux=True."""

    @custom_jvp
    def fn_with_aux(x):
        return ops.multiply(x, 2.0), "aux_data"

    @fn_with_aux.defjvp
    def fn_with_aux_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        return (ops.multiply(x, 2.0), "aux_data"), ops.multiply(t, 99.0)

    x = Tensor(np.array([5.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    (val, aux), tan = jvp(fn_with_aux, (x,), (t,), has_aux=True)

    assert aux == "aux_data"
    np.testing.assert_allclose(get_active_backend().asarray(val), np.array([10.0], dtype=np.float32))
    np.testing.assert_allclose(get_active_backend().asarray(tan), np.array([99.0], dtype=np.float32))


def test_custom_jvp_without_rule_fallback():
    """Verify fallback behavior when no custom JVP rule is registered."""

    @custom_jvp
    def standard_fn(x):
        return ops.multiply(x, 4.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = jvp(standard_fn, (x,), (t,))

    np.testing.assert_allclose(get_active_backend().asarray(val), np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(get_active_backend().asarray(tan), np.array([4.0], dtype=np.float32))


def test_custom_jvp_equality_and_properties():
    """Verify CustomJVPFunction equality, representation, and non-tensor call."""

    def dummy(a, b):
        return a + b

    cf1 = CustomJVPFunction(dummy)
    cf2 = CustomJVPFunction(dummy)

    def dummy2(a, b):
        return a - b

    cf3 = CustomJVPFunction(dummy2)

    assert cf1 == cf2
    assert cf1 == dummy
    assert cf1 != cf3
    assert cf1 != "not_a_fn"

    # Call with non-tensor args
    assert cf1(1, 2) == 3


def test_load_primitive_autodiff_rules_cache_and_path(tmp_path):
    """Verify load_primitive_vjp_rules and load_primitive_jvp_rules loading and caching."""
    # Test cached lookup
    vjp_rules_cached = load_primitive_vjp_rules()
    assert "Add" in vjp_rules_cached
    assert load_primitive_vjp_rules() is vjp_rules_cached

    jvp_rules_cached = load_primitive_jvp_rules()
    assert "Add" in jvp_rules_cached
    assert load_primitive_jvp_rules() is jvp_rules_cached

    # Test loading from explicit custom file path
    dummy_yaml = tmp_path / "custom_vjps.yaml"
    dummy_yaml.write_text("CustomOp:\n  vjp:\n  - $cotangent\n")
    custom_vjp = load_primitive_vjp_rules(str(dummy_yaml))
    assert "CustomOp" in custom_vjp

    dummy_jvp_yaml = tmp_path / "custom_jvps.yaml"
    dummy_jvp_yaml.write_text("CustomOp:\n  jvp: $tangent[0]\n")
    custom_jvp_res = load_primitive_jvp_rules(str(dummy_jvp_yaml))
    assert "CustomOp" in custom_jvp_res

    # Test has_vjp and has_jvp
    assert has_vjp("Add") is True
    assert has_jvp("Add") is True
    assert has_vjp("NonExistentOpXYZ") is False
    assert "NonExistentOpXYZ" not in load_primitive_jvp_rules()


def test_custom_jvp_vjp_fallback_branch():
    """Verify custom_jvp_vjp fallback path when fun is not in attributes or node.inputs is empty."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    g = LogicalGraph(name="test_g")
    dummy_node = LogicalNode(
        id="cjvp_dummy",
        op_type="CustomJVP",
        inputs=["in1"],
        attributes={},
        shape_metadata=(),
    )
    res = custom_jvp_vjp(g, dummy_node, "cot_id")
    assert res == ("cot_id",)


def test_custom_jvp_and_vjp_tracing_and_unpacked_jvp():
    """Test lines 82, 98-108, 163-170, and 312-347 in grad/jvp_vjp.py."""
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.jvp_vjp import custom_jvp, jvp, vjp
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    @custom_jvp
    def my_fn(x):
        return x * 2

    @my_fn.defjvp
    def my_jvp(primals, tangents):
        p = primals[0]
        t = tangents[0]
        return p * 2, t * 2

    # Line 82 & 98-108: tracing active
    g = LogicalGraph("trace_g")
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = g
    try:
        config.eager_mode = False
        # Line 82: non-tensor arg
        assert my_fn(5.0) == 10.0

        # Lines 98-108: Tensor arg with data lacking id
        t_noid = Tensor(5.0, TensorConfig((), DType.Float32, Device("cpu")))
        res_t = my_fn(t_noid)
        assert res_t is not None
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
        config.eager_mode = True

    # Lines 163-164 & 169-170: unpacked signature and single return
    @custom_jvp
    def unpack_fn(x):
        return x + 1

    def unpack_jvp(p, t):
        return t

    unpack_fn.defjvp(unpack_jvp)
    val, tan = jvp(unpack_fn, (2.0,), (1.0,))
    assert val == 3.0
    assert tan == (1.0,)

    @custom_jvp
    def unpack4_fn(x, y):
        return x + y

    def unpack4_jvp(p1, p2, t1, t2):
        return (p1 + p2, t1 + t2)

    unpack4_fn.defjvp(unpack4_jvp)
    val4, tan4 = jvp(unpack4_fn, (2.0, 3.0), (1.0, 1.0))
    assert val4 == 5.0
    assert tan4 == 2.0

    # Lines 312-347: vjp_fn called during active graph tracing
    def simple_f(x):
        return x * 3.0

    _, vjp_fn = vjp(simple_f, 2.0)
    g_active = LogicalGraph("g_active")
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = g_active
    try:
        cot = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))
        grads = vjp_fn(cot)
        assert len(grads) == 1
    finally:
        global_tracing_state.is_tracing = False
        global_tracing_state.active_graph = None
