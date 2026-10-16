"""Unit tests for custom JVP implementation and forward-mode autodiff integration."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import numpy as np

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import custom_jvp, grad, jvp
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_jvp_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import (
    has_jvp,
    load_primitive_jvp_rules,
)
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import (
    has_vjp,
    load_primitive_vjp_rules,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

CustomJVPFunction = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.grad"), "CustomJVPFunction"))


def test_custom_jvp_decorator_and_eager() -> None:
    """Verify custom_jvp decorator and eager evaluation with custom rule."""

    @custom_jvp
    def square_scaled(x: EagerValue) -> EagerValue:
        """Test function square_scaled."""
        return ops.multiply(x, x)

    @square_scaled.defjvp
    def square_scaled_jvp(primals: EagerValue, tangents: EagerValue) -> EagerValue:
        """Test function square_scaled_jvp."""
        (x,) = cast("EagerTensorProtocol", primals)
        (t,) = cast("EagerTensorProtocol", tangents)
        # Custom derivative: instead of 2 * x * t, use 42 * x * t
        primal_out = square_scaled(x)
        tan_out = ops.multiply(ops.multiply(x, 42.0), t)
        return primal_out, tan_out

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = cast("Callable[..., EagerTensorProtocol]", jvp)(square_scaled, (x,), (t,))

    val_arr = get_active_backend().asarray(val)
    tan_arr = get_active_backend().asarray(tan)

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(val_arr, cast("Callable[..., EagerTensorProtocol]", np.array)([4.0], dtype=np.float32))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(tan_arr, cast("Callable[..., EagerTensorProtocol]", np.array)([84.0], dtype=np.float32))


def test_custom_jvp_graph_integration() -> None:
    """Verify custom JVP is properly handled within an outer traced graph."""

    @custom_jvp
    def cube_custom(x: EagerValue) -> EagerValue:
        """Test function cube_custom."""
        return ops.multiply(x, ops.multiply(x, x))

    @cube_custom.defjvp
    def cube_custom_jvp(primals: EagerValue, tangents: EagerValue) -> EagerValue:
        """Test function cube_custom_jvp."""
        (x,) = cast("EagerTensorProtocol", primals)
        (t,) = cast("EagerTensorProtocol", tangents)
        # Override derivative to 10.0 * t
        return cube_custom(x), ops.multiply(t, 10.0)

    def outer(x: EagerValue) -> EagerValue:
        """Test function outer."""
        return ops.add(cube_custom(x), 5.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = cast("Callable[..., EagerTensorProtocol]", jvp)(outer, (x,), (t,))

    tan_arr = get_active_backend().asarray(tan)
    # Derivative of outer w.r.t x should be 10.0 (from custom JVP rule)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(tan_arr, cast("Callable[..., EagerTensorProtocol]", np.array)([10.0], dtype=np.float32))


def test_custom_jvp_reverse_mode_grad() -> None:
    """Verify reverse-mode grad correctly propagates through CustomJVP nodes."""

    @custom_jvp
    def f_custom(x: EagerValue) -> EagerValue:
        """Test function f_custom."""
        return ops.multiply(x, x)

    @f_custom.defjvp
    def f_custom_jvp(primals: EagerValue, tangents: EagerValue) -> EagerValue:
        """Test function f_custom_jvp."""
        (x,) = cast("EagerTensorProtocol", primals)
        (t,) = cast("EagerTensorProtocol", tangents)
        return f_custom(x), ops.multiply(t, 2.0)

    def outer(x: EagerValue) -> EagerValue:
        """Test function outer."""
        return f_custom(x)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    res_grad = grad(outer)(x)

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(res_grad), cast("Callable[..., EagerTensorProtocol]", np.array)([6.0], dtype=np.float32))


def test_custom_jvp_has_aux() -> None:
    """Verify custom JVP with has_aux=True."""

    @custom_jvp
    def fn_with_aux(x: EagerValue) -> EagerValue:
        """Test function fn_with_aux."""
        return ops.multiply(x, 2.0), "aux_data"

    @fn_with_aux.defjvp
    def fn_with_aux_jvp(primals: EagerValue, tangents: EagerValue) -> EagerValue:
        """Test function fn_with_aux_jvp."""
        (x,) = cast("EagerTensorProtocol", primals)
        (t,) = cast("EagerTensorProtocol", tangents)
        return (ops.multiply(x, 2.0), "aux_data"), ops.multiply(t, 99.0)

    x = Tensor(np.array([5.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    (val, aux), tan = cast("Callable[..., EagerTensorProtocol]", jvp)(fn_with_aux, (x,), (t,), has_aux=True)

    assert aux == "aux_data"
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(val), cast("Callable[..., EagerTensorProtocol]", np.array)([10.0], dtype=np.float32))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(tan), cast("Callable[..., EagerTensorProtocol]", np.array)([99.0], dtype=np.float32))


def test_custom_jvp_without_rule_fallback() -> None:
    """Verify fallback behavior when no custom JVP rule is registered."""

    @custom_jvp
    def standard_fn(x: EagerValue) -> EagerValue:
        """Test function standard_fn."""
        return ops.multiply(x, 4.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = cast("Callable[..., EagerTensorProtocol]", jvp)(standard_fn, (x,), (t,))

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(val), cast("Callable[..., EagerTensorProtocol]", np.array)([12.0], dtype=np.float32))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(tan), cast("Callable[..., EagerTensorProtocol]", np.array)([4.0], dtype=np.float32))


def test_custom_jvp_equality_and_properties() -> None:
    """Verify CustomJVPFunction equality, representation, and non-tensor call."""

    def dummy(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function dummy."""
        return cast("EagerTensorProtocol", a) + cast("EagerTensorProtocol", b)

    cf1 = CustomJVPFunction(dummy)
    cf2 = CustomJVPFunction(dummy)

    def dummy2(a: EagerValue, b: EagerValue) -> EagerValue:
        """Test function dummy2."""
        return cast("EagerTensorProtocol", a) - cast("EagerTensorProtocol", b)

    cf3 = CustomJVPFunction(dummy2)

    assert cf1 == cf2
    assert cf1 == dummy
    assert cf1 != cf3
    assert cf1 != "not_a_fn"

    # Call with non-tensor args
    assert cf1(1, 2) == 3


def test_load_primitive_autodiff_rules_cache_and_path(tmp_path: EagerTensorProtocol) -> None:
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


def test_custom_jvp_vjp_fallback_branch() -> None:
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


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.jvp_vjp import custom_jvp, jvp, vjp
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    @custom_jvp
    def my_fn(x: EagerValue) -> EagerValue:
        """Test function my_fn."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2)

    @my_fn.defjvp
    def my_jvp(primals: EagerValue, tangents: EagerValue) -> EagerValue:
        """Test function my_jvp."""
        p = cast("EagerTensorProtocol", getattr(primals, "__getitem__")(0))
        t = cast("EagerTensorProtocol", getattr(tangents, "__getitem__")(0))
        return p * 2, t * 2

    # Line 82 & 98-108: tracing active
    g = LogicalGraph("trace_g")
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = g
    getattr(_state, "update")({"DType": DType, "Device": Device, "LogicalGraph": LogicalGraph, "Tensor": Tensor})
    getattr(_state, "update")({"TensorConfig": TensorConfig, "config": config, "custom_jvp": custom_jvp, "g": g})
    getattr(_state, "update")({"global_tracing_state": global_tracing_state, "jvp": jvp, "my_fn": my_fn, "my_jvp": my_jvp})
    getattr(_state, "update")({"vjp": vjp})
    _state.update(locals())


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    DType, Device, Tensor, TensorConfig = _state["DType"], _state["Device"], _state["Tensor"], _state["TensorConfig"]
    config, custom_jvp, global_tracing_state, jvp = _state["config"], _state["custom_jvp"], _state["global_tracing_state"], _state["jvp"]
    my_fn = _state["my_fn"]
    try:
        setattr(config, "eager_mode", False)
        # Line 82: non-tensor arg
        assert my_fn(5.0) == 10.0

        # Lines 98-108: Tensor arg with data lacking id
        t_noid = Tensor(5.0, TensorConfig((), DType.Float32, Device("cpu")))
        res_t = my_fn(t_noid)
        assert res_t is not None
    finally:
        setattr(global_tracing_state, "is_tracing", False)
        setattr(global_tracing_state, "active_graph", None)
        setattr(config, "eager_mode", True)

    # Lines 163-164 & 169-170: unpacked signature and single return
    @custom_jvp
    def unpack_fn(x: EagerValue) -> EagerValue:
        """Test function unpack_fn."""
        return cast("EagerTensorProtocol", x) + cast("EagerTensorProtocol", 1)

    def unpack_jvp(p: EagerValue, t: EagerValue) -> EagerValue:
        """Test function unpack_jvp."""
        return t

    unpack_fn.defjvp(unpack_jvp)
    val, tan = jvp(unpack_fn, (2.0,), (1.0,))
    assert val == 3.0
    assert tan == (1.0,)

    @custom_jvp
    def unpack4_fn(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function unpack4_fn."""
        return cast("EagerTensorProtocol", x) + cast("EagerTensorProtocol", y)

    getattr(_state, "update")({"tan": tan, "unpack4_fn": unpack4_fn, "unpack_fn": unpack_fn, "unpack_jvp": unpack_jvp})
    getattr(_state, "update")({"val": val})
    _state.update(locals())


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    DType, Device, LogicalGraph, Tensor = _state["DType"], _state["Device"], _state["LogicalGraph"], _state["Tensor"]
    TensorConfig, global_tracing_state, jvp, unpack4_fn = _state["TensorConfig"], _state["global_tracing_state"], _state["jvp"], _state["unpack4_fn"]
    vjp = _state["vjp"]

    def unpack4_jvp(p1: EagerValue, p2: EagerValue, t1: EagerValue, t2: EagerValue) -> EagerValue:
        """Test function unpack4_jvp."""
        return (cast("EagerTensorProtocol", p1) + cast("EagerTensorProtocol", p2), cast("EagerTensorProtocol", t1) + cast("EagerTensorProtocol", t2))

    unpack4_fn.defjvp(unpack4_jvp)
    val4, tan4 = jvp(unpack4_fn, (2.0, 3.0), (1.0, 1.0))
    assert val4 == 5.0
    assert tan4 == 2.0

    # Lines 312-347: vjp_fn called during active graph tracing
    def simple_f(x: EagerValue) -> EagerValue:
        """Test function simple_f."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 3.0)

    _, vjp_fn = vjp(simple_f, 2.0)
    g_active = LogicalGraph("g_active")
    setattr(global_tracing_state, "is_tracing", True)
    setattr(global_tracing_state, "active_graph", g_active)
    try:
        cot = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))
        grads = vjp_fn(cot)
        assert len(grads) == 1
    finally:
        setattr(global_tracing_state, "is_tracing", False)
        setattr(global_tracing_state, "active_graph", None)
    getattr(_state, "update")({"_": _, "g_active": g_active, "simple_f": simple_f, "tan4": tan4})
    getattr(_state, "update")({"unpack4_jvp": unpack4_jvp, "val4": val4, "vjp_fn": vjp_fn})


def test_custom_jvp_and_vjp_tracing_and_unpacked_jvp() -> None:
    """Test lines 82, 98-108, 163-170, and 312-347 in grad/jvp_vjp.py."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_1(_state)
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_2(_state)
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_3(_state)
    assert isinstance(_state, dict)


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_ir import LogicalGraph

    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.device import Device
    from ml_switcheroo_compiler.core.dtype import DType
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
    from ml_switcheroo_compiler.grad.custom_vjp_ops import custom_vjp
    from ml_switcheroo_compiler.grad.jvp_vjp import custom_jvp
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # 1. Test custom_jvp defjvp branches:
    subgraph_jvp = LogicalGraph("jvp_sub")

    @custom_jvp
    def fn_jvp_sub(x: EagerValue) -> EagerValue:
        """Test function fn_jvp_sub."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2)

    fn_jvp_sub.defjvp(lambda p, t: (p[0] * 2, t[0] * 2), jvp_subgraph=subgraph_jvp)
    assert fn_jvp_sub.jvp_subgraph is subgraph_jvp

    @custom_jvp
    def fn_jvp_rule_graph(x: EagerValue) -> EagerValue:
        """Test function fn_jvp_rule_graph."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2)

    graph_rule = LogicalGraph("graph_rule")
    cast("Callable[..., EagerTensorProtocol]", fn_jvp_rule_graph.defjvp)(graph_rule)
    assert fn_jvp_rule_graph.jvp_subgraph is graph_rule
    getattr(_state, "update")({"DType": DType, "Device": Device, "LogicalGraph": LogicalGraph, "Tensor": Tensor})
    getattr(_state, "update")({"TensorConfig": TensorConfig, "config": config, "custom_jvp": custom_jvp, "custom_vjp": custom_vjp})
    getattr(_state, "update")({"fn_jvp_rule_graph": fn_jvp_rule_graph, "fn_jvp_sub": fn_jvp_sub, "global_tracing_state": global_tracing_state, "graph_rule": graph_rule})
    getattr(_state, "update")({"subgraph_jvp": subgraph_jvp})
    _state.update(locals())


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    LogicalGraph, custom_jvp, global_tracing_state, subgraph_jvp = _state["LogicalGraph"], _state["custom_jvp"], _state["global_tracing_state"], _state["subgraph_jvp"]

    class RuleWithGraph:
        """Test class RuleWithGraph."""

        graph = LogicalGraph("rule_with_graph")

        def __call__(self, primals: EagerValue, tangents: EagerValue) -> EagerValue:
            """Test function __call__."""
            return cast("EagerTensorProtocol", getattr(primals, "__getitem__")(0)), cast("EagerTensorProtocol", getattr(tangents, "__getitem__")(0))

    @custom_jvp
    def fn_jvp_hasattr_graph(x: EagerValue) -> EagerValue:
        """Test function fn_jvp_hasattr_graph."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 2)

    fn_jvp_hasattr_graph.defjvp(RuleWithGraph())
    assert fn_jvp_hasattr_graph.jvp_subgraph is RuleWithGraph.graph

    fn_jvp_hasattr_graph.defjvp_subgraph(subgraph_jvp)
    assert fn_jvp_hasattr_graph.jvp_subgraph is subgraph_jvp

    # Trace custom_jvp with jvp_subgraph not None
    g_trace_jvp = LogicalGraph("trace_jvp")
    setattr(global_tracing_state, "is_tracing", True)
    setattr(global_tracing_state, "active_graph", g_trace_jvp)
    getattr(_state, "update")({"RuleWithGraph": RuleWithGraph, "fn_jvp_hasattr_graph": fn_jvp_hasattr_graph, "g_trace_jvp": g_trace_jvp})
    _state.update(locals())


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    DType, Device, LogicalGraph, Tensor = _state["DType"], _state["Device"], _state["LogicalGraph"], _state["Tensor"]
    TensorConfig, config, custom_vjp, fn_jvp_sub = _state["TensorConfig"], _state["config"], _state["custom_vjp"], _state["fn_jvp_sub"]
    g_trace_jvp, global_tracing_state, subgraph_jvp = _state["g_trace_jvp"], _state["global_tracing_state"], _state["subgraph_jvp"]
    try:
        setattr(config, "eager_mode", False)
        t = Tensor(2.0, TensorConfig((), DType.Float32, Device("cpu")))
        out = fn_jvp_sub(t)
        assert out is not None
        assert any(n.op_type == "CustomJVP" and n.attributes.get("jvp_graph") is subgraph_jvp for n in g_trace_jvp.nodes.values())
    finally:
        setattr(global_tracing_state, "is_tracing", False)
        setattr(global_tracing_state, "active_graph", None)
        setattr(config, "eager_mode", True)
    subgraph_vjp = LogicalGraph("vjp_sub")

    @custom_vjp
    def fn_vjp(x: EagerValue) -> EagerValue:
        """Test function fn_vjp."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 3)

    fn_vjp.defvjp(lambda x: (x * 3, x), lambda res, cot: (cot * 3,), bwd_subgraph=subgraph_vjp)
    assert fn_vjp.bwd_subgraph is subgraph_vjp

    @custom_vjp
    def fn_vjp_graph(x: EagerValue) -> EagerValue:
        """Test function fn_vjp_graph."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 3)

    fn_vjp_graph.defvjp(lambda x: (x * 3, x), subgraph_vjp)
    assert fn_vjp_graph.bwd_subgraph is subgraph_vjp
    getattr(_state, "update")({"fn_vjp": fn_vjp, "fn_vjp_graph": fn_vjp_graph, "subgraph_vjp": subgraph_vjp})
    _state.update(locals())


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    LogicalGraph, custom_vjp, global_tracing_state, subgraph_vjp = _state["LogicalGraph"], _state["custom_vjp"], _state["global_tracing_state"], _state["subgraph_vjp"]

    class BwdWithGraph:
        """Test class BwdWithGraph."""

        graph = LogicalGraph("bwd_with_graph")

        def __call__(self, *args: EagerValue) -> EagerValue:
            """Test function __call__."""
            return args

    @custom_vjp
    def fn_vjp_hasattr(x: EagerValue) -> EagerValue:
        """Test function fn_vjp_hasattr."""
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 3)

    fn_vjp_hasattr.defvjp(lambda x: (x * 3, x), BwdWithGraph())
    assert fn_vjp_hasattr.bwd_subgraph is BwdWithGraph.graph

    fn_vjp_hasattr.defvjp_subgraph(subgraph_vjp)
    assert fn_vjp_hasattr.bwd_subgraph is subgraph_vjp

    # Trace custom_vjp with bwd_subgraph and Tensor lacking data.id
    g_trace_vjp = LogicalGraph("trace_vjp")
    setattr(global_tracing_state, "is_tracing", True)
    setattr(global_tracing_state, "active_graph", g_trace_vjp)
    getattr(_state, "update")({"BwdWithGraph": BwdWithGraph, "fn_vjp_hasattr": fn_vjp_hasattr, "g_trace_vjp": g_trace_vjp})
    _state.update(locals())


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 5 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    DType, Device, Tensor, TensorConfig = _state["DType"], _state["Device"], _state["Tensor"], _state["TensorConfig"]
    config, custom_vjp, fn_vjp, g_trace_vjp = _state["config"], _state["custom_vjp"], _state["fn_vjp"], _state["g_trace_vjp"]
    global_tracing_state, subgraph_vjp = _state["global_tracing_state"], _state["subgraph_vjp"]
    try:
        setattr(config, "eager_mode", False)
        t_noid = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
        out_v = fn_vjp(t_noid)
        assert out_v is not None
        assert any(n.op_type == "CustomVJP" and n.attributes.get("bwd_graph") is subgraph_vjp for n in g_trace_vjp.nodes.values())
        assert any(n.op_type == "Constant" for n in g_trace_vjp.nodes.values())

        # Trace custom_vjp without bwd_subgraph to exercise lines 150 and 152 branches
        @custom_vjp
        def fn_vjp_nosub(x: EagerValue) -> EagerValue:
            """Test function fn_vjp_nosub."""
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 4)

        fn_vjp_nosub.defvjp(lambda x: (x * 4, x), lambda res, cot: (cot * 4,))
        out_nosub = fn_vjp_nosub(t_noid)
        assert out_nosub is not None
        assert any(n.op_type == "CustomVJP" and "bwd" not in n.subgraphs and "fwd" in n.subgraphs for n in g_trace_vjp.nodes.values())

        # Trace custom_vjp without fwd defined (fwd is None)
        @custom_vjp
        def fn_vjp_nofwd(x: EagerValue) -> EagerValue:
            """Test function fn_vjp_nofwd."""
            return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", 5)

        out_nofwd = fn_vjp_nofwd(t_noid)
        assert out_nofwd is not None
        assert any(n.op_type == "CustomVJP" and "fwd" not in n.subgraphs for n in g_trace_vjp.nodes.values())

        # Test _emit_vjp_node directly with primal_graph=None
        out_no_primal = fn_vjp_nofwd._emit_vjp_node([t_noid], None, None)
        assert out_no_primal is not None
    finally:
        setattr(global_tracing_state, "is_tracing", False)
        setattr(global_tracing_state, "active_graph", None)
        setattr(config, "eager_mode", True)


def test_custom_jvp_and_custom_vjp_subgraph_branches() -> None:
    """Verify subgraph assignment and tracing in custom_jvp and custom_vjp."""
    _state: dict[str, EagerTensorProtocol] = {}
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_1(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_2(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_3(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_4(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_5(_state)
    assert isinstance(_state, dict)
