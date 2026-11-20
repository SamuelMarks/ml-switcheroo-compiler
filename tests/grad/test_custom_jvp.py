"""Unit tests for custom JVP implementation and forward-mode autodiff integration."""

from __future__ import annotations

import importlib
import typing
from pathlib import Path

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import custom_jvp, grad, jvp
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import custom_jvp_vjp
from ml_switcheroo_compiler.transforms.autodiff_rules.jvp_registry import has_jvp, load_primitive_jvp_rules
from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import has_vjp, load_primitive_vjp_rules
from tests.testing_utils import coerce_numeric, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg
    from ml_switcheroo_compiler.core.config import Config
    from ml_switcheroo_compiler.grad.custom_vjp_ops import CustomVJPFunction
    from ml_switcheroo_compiler.grad.jvp_vjp import CustomJVPFunction as _CustomJVPFunctionType
    from ml_switcheroo_compiler.tracing.state import TracingState

CustomJVPFunction = importlib.import_module("ml_switcheroo_compiler.grad").CustomJVPFunction


def test_custom_jvp_decorator_and_eager() -> None:
    """Verify custom_jvp decorator and eager evaluation with custom rule."""

    @custom_jvp
    def square_scaled(x: BackendArg) -> BackendArg:
        """Test function square_scaled.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `square_scaled`.
        """
        return ops.multiply(x, x)

    @square_scaled.defjvp
    def square_scaled_jvp(primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]) -> tuple[BackendArg, BackendArg]:
        """Test function square_scaled_jvp.

        Args:
            primals: Test parameter `primals`.
            tangents: Test parameter `tangents`.

        Returns:
            tuple[BackendArg, BackendArg]: Computed output value from `square_scaled_jvp`.
        """
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

    np.testing.assert_allclose(ensure_ndarray(val_arr), np.array([4.0], dtype=np.float32))
    np.testing.assert_allclose(ensure_ndarray(tan_arr), np.array([84.0], dtype=np.float32))


def test_custom_jvp_graph_integration() -> None:
    """Verify custom JVP is properly handled within an outer traced graph."""

    @custom_jvp
    def cube_custom(x: BackendArg) -> BackendArg:
        """Test function cube_custom.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `cube_custom`.
        """
        return ops.multiply(x, ops.multiply(x, x))

    @cube_custom.defjvp
    def cube_custom_jvp(primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]) -> tuple[BackendArg, BackendArg]:
        """Test function cube_custom_jvp.

        Args:
            primals: Test parameter `primals`.
            tangents: Test parameter `tangents`.

        Returns:
            tuple[BackendArg, BackendArg]: Computed output value from `cube_custom_jvp`.
        """
        (x,) = primals
        (t,) = tangents
        # Override derivative to 10.0 * t
        return cube_custom(x), ops.multiply(t, 10.0)

    def outer(x: BackendArg) -> BackendArg:
        """Test function outer.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `outer`.
        """
        return ops.add(cube_custom(x), 5.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = jvp(outer, (x,), (t,))

    tan_arr = get_active_backend().asarray(tan)
    # Derivative of outer w.r.t x should be 10.0 (from custom JVP rule)
    np.testing.assert_allclose(ensure_ndarray(tan_arr), np.array([10.0], dtype=np.float32))


def test_custom_jvp_reverse_mode_grad() -> None:
    """Verify reverse-mode grad correctly propagates through CustomJVP nodes."""

    @custom_jvp
    def f_custom(x: BackendArg) -> BackendArg:
        """Test function f_custom.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f_custom`.
        """
        return ops.multiply(x, x)

    @f_custom.defjvp
    def f_custom_jvp(primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]) -> tuple[BackendArg, BackendArg]:
        """Test function f_custom_jvp.

        Args:
            primals: Test parameter `primals`.
            tangents: Test parameter `tangents`.

        Returns:
            tuple[BackendArg, BackendArg]: Computed output value from `f_custom_jvp`.
        """
        (x,) = primals
        (t,) = tangents
        return f_custom(x), ops.multiply(t, 2.0)

    def outer(x: BackendArg) -> BackendArg:
        """Test function outer.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `outer`.
        """
        return f_custom(x)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    res_grad = grad(outer)(x)

    np.testing.assert_allclose(
        ensure_ndarray(get_active_backend().asarray(res_grad)), np.array([6.0], dtype=np.float32)
    )


def test_custom_jvp_has_aux() -> None:
    """Verify custom JVP with has_aux=True."""

    @custom_jvp
    def fn_with_aux(x: BackendArg) -> BackendArg:
        """Test function fn_with_aux.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn_with_aux`.
        """
        return ops.multiply(x, 2.0), "aux_data"

    @fn_with_aux.defjvp
    def fn_with_aux_jvp(primals: tuple[Tensor, ...], tangents: tuple[Tensor, ...]) -> tuple[BackendArg, BackendArg]:
        """Test function fn_with_aux_jvp.

        Args:
            primals: Test parameter `primals`.
            tangents: Test parameter `tangents`.

        Returns:
            tuple[BackendArg, BackendArg]: Computed output value from `fn_with_aux_jvp`.
        """
        (x,) = primals
        (t,) = tangents
        return (ops.multiply(x, 2.0), "aux_data"), ops.multiply(t, 99.0)

    x = Tensor(np.array([5.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val_aux, tan = unpack_pair(jvp(fn_with_aux, (x,), (t,), has_aux=True))
    val, aux = unpack_pair(val_aux)

    assert aux == "aux_data"
    np.testing.assert_allclose(ensure_ndarray(get_active_backend().asarray(val)), np.array([10.0], dtype=np.float32))
    np.testing.assert_allclose(ensure_ndarray(get_active_backend().asarray(tan)), np.array([99.0], dtype=np.float32))


def test_custom_jvp_without_rule_fallback() -> None:
    """Verify fallback behavior when no custom JVP rule is registered."""

    @custom_jvp
    def standard_fn(x: BackendArg) -> BackendArg:
        """Test function standard_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `standard_fn`.
        """
        return ops.multiply(x, 4.0)

    x = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    t = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, tan = jvp(standard_fn, (x,), (t,))

    np.testing.assert_allclose(ensure_ndarray(get_active_backend().asarray(val)), np.array([12.0], dtype=np.float32))
    np.testing.assert_allclose(ensure_ndarray(get_active_backend().asarray(tan)), np.array([4.0], dtype=np.float32))


def test_custom_jvp_equality_and_properties() -> None:
    """Verify CustomJVPFunction equality, representation, and non-tensor call."""

    def dummy(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function dummy.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `dummy`.
        """
        return coerce_numeric(a) + coerce_numeric(b)

    cf1 = CustomJVPFunction(dummy)
    cf2 = CustomJVPFunction(dummy)

    def dummy2(a: BackendArg, b: BackendArg) -> BackendArg:
        """Test function dummy2.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            BackendArg: Computed output value from `dummy2`.
        """
        return coerce_numeric(a) - coerce_numeric(b)

    cf3 = CustomJVPFunction(dummy2)

    assert cf1 == cf2
    assert cf1 == dummy
    assert cf1 != cf3
    assert cf1 != "not_a_fn"

    # Call with non-tensor args
    assert cf1(1, 2) == 3


def test_load_primitive_autodiff_rules_cache_and_path(tmp_path: Path) -> None:
    """Verify load_primitive_vjp_rules and load_primitive_jvp_rules loading and caching.

    Args:
        tmp_path: Test parameter `tmp_path`.
    """
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


class _TestCustomJvpAndVjpTracingAndUnpackedJvpState(typing.TypedDict, total=False):
    """Typed shared state for `_test_custom_jvp_and_vjp_tracing_and_unpacked_jvp` split helpers."""

    config: Config
    global_tracing_state: TracingState
    my_fn: _CustomJVPFunctionType
    unpack4_fn: _CustomJVPFunctionType


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_1(
    _state: _TestCustomJvpAndVjpTracingAndUnpackedJvpState,
) -> None:
    """Execute part 1 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    @custom_jvp
    def my_fn(x: BackendArg) -> BackendArg:
        """Test function my_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_fn`.
        """
        return coerce_numeric(x) * 2

    def my_jvp(primals: BackendArg, tangents: BackendArg) -> BackendArg:
        """Test function my_jvp.

        Args:
            primals: Test parameter `primals`.
            tangents: Test parameter `tangents`.

        Returns:
            BackendArg: Computed output value from `my_jvp`.
        """
        p = ensure_ndarray(primals)[0]
        t = ensure_ndarray(tangents)[0]
        return coerce_numeric(p) * 2, coerce_numeric(t) * 2

    my_fn.defjvp(my_jvp)

    # Line 82 & 98-108: tracing active
    g = LogicalGraph("trace_g")
    global_tracing_state.is_tracing = True
    global_tracing_state.active_graph = g
    _state.update({"config": config, "global_tracing_state": global_tracing_state, "my_fn": my_fn})


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_2(
    _state: _TestCustomJvpAndVjpTracingAndUnpackedJvpState,
) -> None:
    """Execute part 2 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    config, global_tracing_state = _state["config"], _state["global_tracing_state"]
    my_fn = _state["my_fn"]
    try:
        pytest.MonkeyPatch().setattr(config, "eager_mode", False, raising=False)
        # Line 82: non-tensor arg
        assert my_fn(5.0) == 10.0

        # Lines 98-108: Tensor arg with data lacking id
        t_noid = Tensor(5.0, TensorConfig((), DType.Float32, Device("cpu")))
        res_t = my_fn(t_noid)
        assert res_t is not None
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", None, raising=False)
        pytest.MonkeyPatch().setattr(config, "eager_mode", True, raising=False)

    # Lines 163-164 & 169-170: unpacked signature and single return
    @custom_jvp
    def unpack_fn(x: BackendArg) -> BackendArg:
        """Test function unpack_fn.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `unpack_fn`.
        """
        return coerce_numeric(x) + 1

    def unpack_jvp(p: BackendArg, t: BackendArg) -> BackendArg:
        """Test function unpack_jvp.

        Args:
            p: Test parameter `p`.
            t: Test parameter `t`.

        Returns:
            BackendArg: Computed output value from `unpack_jvp`.
        """
        return t

    unpack_fn.defjvp(unpack_jvp)
    val, tan = jvp(unpack_fn, (2.0,), (1.0,))
    assert val == 3.0
    assert tan == (1.0,)

    @custom_jvp
    def unpack4_fn(x: BackendArg, y: BackendArg) -> BackendArg:
        """Test function unpack4_fn.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            BackendArg: Computed output value from `unpack4_fn`.
        """
        return coerce_numeric(x) + coerce_numeric(y)

    _state.update({"unpack4_fn": unpack4_fn})


def _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_3(
    _state: _TestCustomJvpAndVjpTracingAndUnpackedJvpState,
) -> None:
    """Execute part 3 of test_custom_jvp_and_vjp_tracing_and_unpacked_jvp.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.grad.jvp_vjp import vjp

    global_tracing_state, unpack4_fn = _state["global_tracing_state"], _state["unpack4_fn"]

    def unpack4_jvp(p1: BackendArg, p2: BackendArg, t1: BackendArg, t2: BackendArg) -> BackendArg:
        """Test function unpack4_jvp.

        Args:
            p1: Test parameter `p1`.
            p2: Test parameter `p2`.
            t1: Test parameter `t1`.
            t2: Test parameter `t2`.

        Returns:
            BackendArg: Computed output value from `unpack4_jvp`.
        """
        return (coerce_numeric(p1) + coerce_numeric(p2), coerce_numeric(t1) + coerce_numeric(t2))

    unpack4_fn.defjvp(unpack4_jvp)
    val4, tan4 = jvp(unpack4_fn, (2.0, 3.0), (1.0, 1.0))
    assert val4 == 5.0
    assert tan4 == 2.0

    # Lines 312-347: vjp_fn called during active graph tracing
    def simple_f(x: Tensor | float) -> Tensor | float:
        """Test function simple_f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor | float: Computed output value from `simple_f`.
        """
        return x * 3.0

    _, vjp_fn = vjp(simple_f, 2.0)
    g_active = LogicalGraph("g_active")
    pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", True, raising=False)
    global_tracing_state.active_graph = g_active
    try:
        cot = Tensor(1.0, TensorConfig((), DType.Float32, Device("cpu")))
        grads = vjp_fn(cot)
        assert len(grads) == 1
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", None, raising=False)


def test_custom_jvp_and_vjp_tracing_and_unpacked_jvp() -> None:
    """Test lines 82, 98-108, 163-170, and 312-347 in grad/jvp_vjp.py."""
    _state: _TestCustomJvpAndVjpTracingAndUnpackedJvpState = {}
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_1(_state)
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_2(_state)
    _test_custom_jvp_and_vjp_tracing_and_unpacked_jvp_part_3(_state)
    assert len(_state) > 0


class _TestCustomJvpAndCustomVjpSubgraphBranchesState(typing.TypedDict, total=False):
    """Typed shared state for `_test_custom_jvp_and_custom_vjp_subgraph_branches` split helpers."""

    config: Config
    fn_jvp_sub: _CustomJVPFunctionType
    fn_vjp: CustomVJPFunction
    g_trace_jvp: LogicalGraph
    g_trace_vjp: LogicalGraph
    global_tracing_state: TracingState
    subgraph_jvp: LogicalGraph
    subgraph_vjp: LogicalGraph


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_1(
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState,
) -> None:
    """Execute part 1 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.tracing.state import global_tracing_state

    # 1. Test custom_jvp defjvp branches:
    subgraph_jvp = LogicalGraph("jvp_sub")

    @custom_jvp
    def fn_jvp_sub(x: BackendArg) -> BackendArg:
        """Test function fn_jvp_sub.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn_jvp_sub`.
        """
        return coerce_numeric(x) * 2

    fn_jvp_sub.defjvp(lambda p, t: (p[0] * 2, t[0] * 2), jvp_subgraph=subgraph_jvp)
    assert fn_jvp_sub.jvp_subgraph is subgraph_jvp

    @custom_jvp
    def fn_jvp_rule_graph(x: BackendArg) -> BackendArg:
        """Test function fn_jvp_rule_graph.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn_jvp_rule_graph`.
        """
        return coerce_numeric(x) * 2

    graph_rule = LogicalGraph("graph_rule")
    fn_jvp_rule_graph.defjvp(graph_rule)
    assert fn_jvp_rule_graph.jvp_subgraph is graph_rule
    _state.update(
        {
            "config": config,
            "fn_jvp_sub": fn_jvp_sub,
            "global_tracing_state": global_tracing_state,
            "subgraph_jvp": subgraph_jvp,
        }
    )


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_2(
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState,
) -> None:
    """Execute part 2 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    global_tracing_state, subgraph_jvp = _state["global_tracing_state"], _state["subgraph_jvp"]

    class RuleWithGraph:
        """Test class RuleWithGraph."""

        graph = LogicalGraph("rule_with_graph")

        def __call__(self, primals: BackendArg, tangents: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                primals: Test parameter `primals`.
                tangents: Test parameter `tangents`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return ensure_ndarray(primals)[0], ensure_ndarray(tangents)[0]

    @custom_jvp
    def fn_jvp_hasattr_graph(x: BackendArg) -> BackendArg:
        """Test function fn_jvp_hasattr_graph.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `fn_jvp_hasattr_graph`.
        """
        return coerce_numeric(x) * 2

    fn_jvp_hasattr_graph.defjvp(RuleWithGraph())
    assert fn_jvp_hasattr_graph.jvp_subgraph is RuleWithGraph.graph

    fn_jvp_hasattr_graph.defjvp_subgraph(subgraph_jvp)
    assert fn_jvp_hasattr_graph.jvp_subgraph is subgraph_jvp

    # Trace custom_jvp with jvp_subgraph not None
    g_trace_jvp = LogicalGraph("trace_jvp")
    pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", True, raising=False)
    global_tracing_state.active_graph = g_trace_jvp
    _state.update({"g_trace_jvp": g_trace_jvp})


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_3(
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState,
) -> None:
    """Execute part 3 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.grad.custom_vjp_ops import custom_vjp

    config, fn_jvp_sub = _state["config"], _state["fn_jvp_sub"]
    g_trace_jvp, global_tracing_state, subgraph_jvp = (
        _state["g_trace_jvp"],
        _state["global_tracing_state"],
        _state["subgraph_jvp"],
    )
    try:
        pytest.MonkeyPatch().setattr(config, "eager_mode", False, raising=False)
        t = Tensor(2.0, TensorConfig((), DType.Float32, Device("cpu")))
        out = fn_jvp_sub(t)
        assert out is not None
        assert any(
            n.op_type == "CustomJVP" and n.attributes.get("jvp_graph") is subgraph_jvp
            for n in g_trace_jvp.nodes.values()
        )
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", None, raising=False)
        pytest.MonkeyPatch().setattr(config, "eager_mode", True, raising=False)
    subgraph_vjp = LogicalGraph("vjp_sub")

    @custom_vjp
    def fn_vjp(x: Tensor | float) -> Tensor | float:
        """Test function fn_vjp.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor | float: Computed output value from `fn_vjp`.
        """
        return x * 3

    fn_vjp.defvjp(lambda x: (x * 3, x), lambda res, cot: (cot * 3,), bwd_subgraph=subgraph_vjp)
    assert fn_vjp.bwd_subgraph is subgraph_vjp

    @custom_vjp
    def fn_vjp_graph(x: Tensor | float) -> Tensor | float:
        """Test function fn_vjp_graph.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor | float: Computed output value from `fn_vjp_graph`.
        """
        return x * 3

    fn_vjp_graph.defvjp(lambda x: (x * 3, x), subgraph_vjp)
    assert fn_vjp_graph.bwd_subgraph is subgraph_vjp
    _state.update({"fn_vjp": fn_vjp, "subgraph_vjp": subgraph_vjp})


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_4(
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState,
) -> None:
    """Execute part 4 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.grad.custom_vjp_ops import custom_vjp

    global_tracing_state, subgraph_vjp = _state["global_tracing_state"], _state["subgraph_vjp"]

    class BwdWithGraph:
        """Test class BwdWithGraph."""

        graph = LogicalGraph("bwd_with_graph")

        def __call__(self, *args: BackendArg) -> BackendArg:
            """Test function __call__.

            Args:
                args: Test parameter `args`.

            Returns:
                BackendArg: Computed output value from `__call__`.
            """
            return args

    @custom_vjp
    def fn_vjp_hasattr(x: Tensor | float) -> Tensor | float:
        """Test function fn_vjp_hasattr.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor | float: Computed output value from `fn_vjp_hasattr`.
        """
        return x * 3

    fn_vjp_hasattr.defvjp(lambda x: (x * 3, x), BwdWithGraph())
    assert fn_vjp_hasattr.bwd_subgraph is BwdWithGraph.graph

    fn_vjp_hasattr.defvjp_subgraph(subgraph_vjp)
    assert fn_vjp_hasattr.bwd_subgraph is subgraph_vjp

    # Trace custom_vjp with bwd_subgraph and Tensor lacking data.id
    g_trace_vjp = LogicalGraph("trace_vjp")
    pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", True, raising=False)
    global_tracing_state.active_graph = g_trace_vjp
    _state.update({"g_trace_vjp": g_trace_vjp})


def _test_custom_jvp_and_custom_vjp_subgraph_branches_part_5(
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState,
) -> None:
    """Execute part 5 of test_custom_jvp_and_custom_vjp_subgraph_branches.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.grad.custom_vjp_ops import custom_vjp

    config, fn_vjp, g_trace_vjp = _state["config"], _state["fn_vjp"], _state["g_trace_vjp"]
    global_tracing_state, subgraph_vjp = _state["global_tracing_state"], _state["subgraph_vjp"]
    try:
        pytest.MonkeyPatch().setattr(config, "eager_mode", False, raising=False)
        t_noid = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
        out_v = fn_vjp(t_noid)
        assert out_v is not None
        assert any(
            n.op_type == "CustomVJP" and n.attributes.get("bwd_graph") is subgraph_vjp
            for n in g_trace_vjp.nodes.values()
        )
        assert any(n.op_type == "Constant" for n in g_trace_vjp.nodes.values())

        # Trace custom_vjp without bwd_subgraph to exercise lines 150 and 152 branches
        @custom_vjp
        def fn_vjp_nosub(x: Tensor | float) -> Tensor | float:
            """Test function fn_vjp_nosub.

            Args:
                x: Test parameter `x`.

            Returns:
                Tensor | float: Computed output value from `fn_vjp_nosub`.
            """
            return x * 4

        fn_vjp_nosub.defvjp(lambda x: (x * 4, x), lambda res, cot: (cot * 4,))
        out_nosub = fn_vjp_nosub(t_noid)
        assert out_nosub is not None
        assert any(
            n.op_type == "CustomVJP" and "bwd" not in n.subgraphs and "fwd" in n.subgraphs
            for n in g_trace_vjp.nodes.values()
        )

        # Trace custom_vjp without fwd defined (fwd is None)
        @custom_vjp
        def fn_vjp_nofwd(x: Tensor | ProxyTensor | float) -> Tensor | ProxyTensor | float:
            """Test function fn_vjp_nofwd.

            Args:
                x: Test parameter `x`.

            Returns:
                Tensor | ProxyTensor | float: Computed output value from `fn_vjp_nofwd`.
            """
            return x * 5

        out_nofwd = fn_vjp_nofwd(t_noid)
        assert out_nofwd is not None
        assert any(n.op_type == "CustomVJP" and "fwd" not in n.subgraphs for n in g_trace_vjp.nodes.values())

        # Test _emit_vjp_node directly with primal_graph=None
        out_no_primal = fn_vjp_nofwd._emit_vjp_node([t_noid], None, None)
        assert out_no_primal is not None
    finally:
        pytest.MonkeyPatch().setattr(global_tracing_state, "is_tracing", False, raising=False)
        pytest.MonkeyPatch().setattr(global_tracing_state, "active_graph", None, raising=False)
        pytest.MonkeyPatch().setattr(config, "eager_mode", True, raising=False)


def test_custom_jvp_and_custom_vjp_subgraph_branches() -> None:
    """Verify subgraph assignment and tracing in custom_jvp and custom_vjp."""
    _state: _TestCustomJvpAndCustomVjpSubgraphBranchesState = {}
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_1(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_2(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_3(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_4(_state)
    _test_custom_jvp_and_custom_vjp_subgraph_branches_part_5(_state)
    assert len(_state) > 0
