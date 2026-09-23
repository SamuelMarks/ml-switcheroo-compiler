"""Comprehensive unit tests covering edge cases and branch coverage across core compiler modules."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_ir.types import DType, TensorSpec

from ml_switcheroo_compiler.backends.llvm_cpp import generator as cpp_gen
from ml_switcheroo_compiler.backends.mapping_loader import (
    _MAPPING_CACHE,
    BackendMappingSchema,
    OpMappingSchema,
    dispatch_eager_op,
)
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.custom_vjp_ops import CustomVJPFunction
from ml_switcheroo_compiler.interpreter.evaluator import _prepare_node_kwargs
from ml_switcheroo_compiler.ops.control_flow_utils import _wrap_proxy_inputs
from ml_switcheroo_compiler.ops.creation.frontend_matrix import diag
from ml_switcheroo_compiler.ops.creation.frontend_utils import _emit_constant_node
from ml_switcheroo_compiler.tracing.state import TracingState, global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_compiler.transforms.passes.dce import (
    _node_has_side_effects,
    _prune_nested_subgraphs,
)
from ml_switcheroo_compiler.transforms.passes.graph_scheduling import GraphSchedulingPass
from ml_switcheroo_compiler.transforms.passes.operator_fusion import (
    FusionRule,
    NodePattern,
    PatternMatchingEngine,
)


def test_evaluator_prepare_node_kwargs_reshape() -> None:
    """Test _prepare_node_kwargs adds shape_metadata as newshape for Reshape node."""
    node: LogicalNode = LogicalNode(
        id="reshape_node",
        op_type="Reshape",
        inputs=["in0"],
        attributes={},
        shape_metadata=(2, 4),
    )
    kwargs: dict[str, object] = _prepare_node_kwargs(node, "Reshape")
    assert kwargs.get("newshape") == (2, 4)

    # Test popping strides for non-conv/pool op
    node_strides: LogicalNode = LogicalNode(
        id="add_strides",
        op_type="Add",
        inputs=["in0"],
        attributes={"strides": [1, 1]},
    )
    kw2: dict[str, object] = _prepare_node_kwargs(node_strides, "Add")
    assert "strides" not in kw2


def test_evaluator_handle_index_put_and_checkpoint() -> None:
    """Test _handle_index_put and _handle_checkpoint in evaluator."""
    from ml_switcheroo_compiler.interpreter.environment import Environment
    from ml_switcheroo_compiler.interpreter.evaluator import _dispatch_op

    env: Environment = Environment()
    node: LogicalNode = LogicalNode(id="out1", op_type="IndexPut", inputs=["in0", "in1"])
    _dispatch_op(node, env, None, "IndexPut", [np.array([1, 2, 3]), 99], {"key": "1"})  # type: ignore[arg-type]
    res: np.ndarray = env.get("out1")  # type: ignore[assignment]
    assert res[1] == 99

    sub: LogicalGraph = LogicalGraph(name="sub_chk")
    sub.inputs = ["s_in"]
    sub.outputs = ["s_in"]
    sub.nodes = {"s_in": LogicalNode(id="s_in", op_type="Input", inputs=[])}
    node2: LogicalNode = LogicalNode(id="out2", op_type="Checkpoint", inputs=["in0"], subgraphs={"body": sub})
    _dispatch_op(node2, env, None, "Checkpoint", ["checkpointed_val"], {})  # type: ignore[arg-type]
    assert env.get("out2") == "checkpointed_val"


def test_tracer_proxy_tensor_active_graph_without_input_specs() -> None:
    """Test ProxyTensor registration with and without input_specs attribute."""

    class MockGraphNoSpecs:
        """Mock graph lacking input_specs attribute."""

        def __init__(self) -> None:
            """Initialize MockGraphNoSpecs."""
            self.nodes: dict[str, object] = {}
            self.inputs: list[str] = []

    old_tracing: bool = global_tracing_state.is_tracing
    old_graph: object = global_tracing_state.active_graph
    try:
        global_tracing_state.is_tracing = True
        # Without input_specs
        mock_g = MockGraphNoSpecs()
        global_tracing_state.active_graph = mock_g  # type: ignore[assignment]
        proxy1: ProxyTensor = ProxyTensor(id="test_node_no_specs", shape=(2, 3), dtype="float32")
        assert proxy1.id in mock_g.inputs

        # With input_specs
        g_with_specs = LogicalGraph(name="with_specs")
        global_tracing_state.active_graph = g_with_specs
        proxy2: ProxyTensor = ProxyTensor(id="test_node_with_specs", shape=(4,), dtype="int32")
        assert proxy2.id in g_with_specs.inputs
        assert proxy2.id in g_with_specs.input_specs
    finally:
        global_tracing_state.is_tracing = old_tracing
        global_tracing_state.active_graph = old_graph  # type: ignore[assignment]


def test_tracing_state_add_node_removes_from_inputs() -> None:
    """Test TracingState.add_node removes non-Input node from active_graph inputs and specs."""
    state: TracingState = TracingState()
    graph: LogicalGraph = LogicalGraph(name="test_graph")
    graph.inputs = ["existing_id", "no_specs_id"]
    graph.input_specs = {"existing_id": TensorSpec(shape=(1,), dtype="float32")}

    state.is_tracing = True
    state.active_graph = graph

    # Adding an Input node (op_type == 'Input') covers the false branch of op_type != 'Input'
    input_node: LogicalNode = LogicalNode(
        id="input_node_id",
        op_type="Input",
        inputs=[],
    )
    state.add_node(input_node)
    assert "input_node_id" in graph.nodes

    node1: LogicalNode = LogicalNode(
        id="existing_id",
        op_type="Add",
        inputs=[],
    )
    state.add_node(node1)
    assert "existing_id" not in graph.inputs
    assert "existing_id" not in graph.input_specs

    # Graph without input_specs attribute
    del graph.input_specs
    node2: LogicalNode = LogicalNode(
        id="no_specs_id",
        op_type="Mul",
        inputs=[],
    )
    state.add_node(node2)
    assert "no_specs_id" not in graph.inputs

    # Test nested start_tracing pushing to graph_stack and stop_tracing restoring it
    nested_state: TracingState = TracingState()
    g1 = nested_state.start_tracing("g1")
    g2 = nested_state.start_tracing("g2")
    assert len(nested_state.graph_stack) == 1
    assert nested_state.active_graph.name == "g2"
    finished_g2 = nested_state.stop_tracing()
    assert finished_g2.name == "g2"
    assert nested_state.active_graph.name == "g1"
    finished_g1 = nested_state.stop_tracing()
    assert finished_g1.name == "g1"
    assert nested_state.active_graph is None


def test_control_flow_utils_wrap_proxy_inputs_no_specs() -> None:
    """Test _wrap_proxy_inputs when subgraph does not have input_specs attribute."""

    class MockSubgraphNoSpecs:
        """Mock subgraph without input_specs."""

        def __init__(self) -> None:
            """Initialize MockSubgraphNoSpecs."""
            self.nodes: dict[str, object] = {}

    subgraph = MockSubgraphNoSpecs()
    t: Tensor = Tensor(np.array([1.0, 2.0], dtype=np.float32), TensorConfig((2,), DType.Float32, "cpu"))
    input_ids, proxy_args = _wrap_proxy_inputs([t], subgraph)
    assert len(input_ids) == 1
    assert len(proxy_args) == 1


def test_frontend_utils_emit_constant_node() -> None:
    """Test _emit_constant_node creates and returns a Tensor under active tracing."""
    old_tracing: bool = global_tracing_state.is_tracing
    old_graph: object = global_tracing_state.active_graph
    try:
        global_tracing_state.is_tracing = True
        g: LogicalGraph = LogicalGraph(name="test_const")
        global_tracing_state.active_graph = g
        res: Tensor = _emit_constant_node(3.14, DType.Float32)
        assert res.shape == ()
        assert res.data.id in g.nodes
    finally:
        global_tracing_state.is_tracing = old_tracing
        global_tracing_state.active_graph = old_graph  # type: ignore[assignment]


def test_frontend_matrix_diag_invalid_shape_tracing() -> None:
    """Test diag raises ValueError when input shape is not 1D or 2D under tracing."""
    old_eager: bool = config.eager_mode
    old_tracing: bool = global_tracing_state.is_tracing
    try:
        config.eager_mode = False
        global_tracing_state.is_tracing = True
        t3d: Tensor = Tensor(np.zeros((2, 2, 2)), TensorConfig((2, 2, 2), DType.Float32, "cpu"))
        with pytest.raises(ValueError, match="diag requires a 1D or 2D tensor"):
            diag(t3d)
    finally:
        config.eager_mode = old_eager
        global_tracing_state.is_tracing = old_tracing


def test_custom_vjp_ops_none_graphs() -> None:
    """Test CustomVJPFunction._emit_vjp_node when primal_graph and fwd_graph are None."""

    def base_fn(x: Tensor) -> Tensor:
        """Evaluate base identity function."""
        return x

    def fwd(x: Tensor) -> tuple[Tensor, Tensor]:
        """Evaluate forward pass."""
        return x, x

    def bwd(res: Tensor, cotangent: Tensor) -> tuple[Tensor]:
        """Evaluate backward pass."""
        return (cotangent,)

    fn: CustomVJPFunction = CustomVJPFunction(base_fn)
    t: Tensor = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, "cpu"))

    # Line 88: _trace_fwd_graph when fwd is None returns None
    assert fn._trace_fwd_graph([t]) is None

    # Branch 108->113: _resolve_output_metadata with empty tensor_args
    meta_empty = fn._resolve_output_metadata([])
    assert meta_empty == ((), "float32", "cpu")

    fn.defvjp(fwd, bwd)

    old_tracing: bool = global_tracing_state.is_tracing
    old_graph: object = global_tracing_state.active_graph
    try:
        global_tracing_state.is_tracing = True
        g: LogicalGraph = LogicalGraph(name="test_vjp")
        global_tracing_state.active_graph = g
        res: Tensor = fn._emit_vjp_node([t], fwd_graph=None, primal_graph=None)
        assert res.shape == (1,)
        assert res.data.id in g.nodes
    finally:
        global_tracing_state.is_tracing = old_tracing
        global_tracing_state.active_graph = old_graph  # type: ignore[assignment]


def test_mapping_loader_dispatch_eager_op_kwarg_map() -> None:
    """Test dispatch_eager_op kwarg_map translation handling."""

    class DummyModule:
        """Dummy backend module with test operations."""

        @staticmethod
        def dummy_fn(x: int, axis: int = 0) -> int:
            """Compute dummy add operation."""
            return x + axis

    op_spec: OpMappingSchema = OpMappingSchema(
        operation="TestDummy",
        backend="mock_backend",
        target_api="dummy_fn",
        kwarg_map={"dim": "axis", "none_dst": None, "existing": "axis"},
        kwarg_translations={"existing": "axis"},
    )
    backend_schema: BackendMappingSchema = BackendMappingSchema(
        backend_name="mock_backend",
        operations={"TestDummy": op_spec},
    )
    _MAPPING_CACHE["mock_backend"] = backend_schema

    res: object = dispatch_eager_op("mock_backend", "TestDummy", (10,), {"dim": 5}, DummyModule)
    assert res == 15


def test_mapping_loader_load_single_file_branches(tmp_path: object) -> None:
    """Test _read_and_merge with operations dict and plain dict."""
    import os

    import yaml

    from ml_switcheroo_compiler.backends.mapping_loader import _read_and_merge

    p1: str = os.path.join(str(tmp_path), "ops1.yaml")
    with open(p1, "w", encoding="utf-8") as f:
        yaml.dump({"operations": {"OpA": {"operation": "OpA"}}}, f)

    p2: str = os.path.join(str(tmp_path), "ops2.yaml")
    with open(p2, "w", encoding="utf-8") as f:
        yaml.dump({"OpB": {"target_api": "test_b"}}, f)

    p3: str = os.path.join(str(tmp_path), "ops3.yaml")
    with open(p3, "w", encoding="utf-8") as f:
        yaml.dump({"operation": "OpC", "target_api": "test_c"}, f)

    d: dict[str, object] = {}
    _read_and_merge(p1, d)
    assert "OpA" in d
    _read_and_merge(p2, d)
    assert "OpB" in d
    _read_and_merge(p3, d)
    assert "OpC" in d


def test_graph_scheduling_multiple_parent_streams() -> None:
    """Test GraphSchedulingPass when a node has parents from multiple distinct streams."""
    graph: LogicalGraph = LogicalGraph(name="test_multi_stream")
    n1: LogicalNode = LogicalNode(id="n1", op_type="Op1", inputs=[])
    n1.stream = "stream_b"
    n2: LogicalNode = LogicalNode(id="n2", op_type="Op2", inputs=[])
    n2.stream = "stream_a"
    n3: LogicalNode = LogicalNode(id="n3", op_type="Merge", inputs=["n1", "n2"])
    graph.nodes = {"n1": n1, "n2": n2, "n3": n3}

    sched_pass: GraphSchedulingPass = GraphSchedulingPass()
    modified: bool = sched_pass.run(graph)
    assert modified
    assert n3.stream == "stream_a"


def test_operator_fusion_graph_without_inputs_attribute() -> None:
    """Test PatternMatchingEngine when graph lacks an inputs attribute during fusion update."""

    class MockGraphNoInputs:
        """Mock graph object lacking inputs attribute."""

        def __init__(self) -> None:
            """Initialize MockGraphNoInputs."""
            self.nodes: dict[str, LogicalNode] = {
                "n1": LogicalNode(id="n1", op_type="Add", inputs=["in_a", "in_b"]),
            }
            self.outputs: list[str] = ["n1"]

    class CustomFusionRule(FusionRule):
        """Custom fusion rule for testing pass execution."""

        def apply(self, graph: object, match: dict[str, object]) -> dict[str, LogicalNode] | None:
            """Apply fusion rewrite."""
            return {"n1": LogicalNode(id="fused", op_type="FusedAdd", inputs=["in_a", "in_b"])}

    graph = MockGraphNoInputs()
    rule = CustomFusionRule("test_rule", NodePattern(op_type="Add"))
    engine: PatternMatchingEngine = PatternMatchingEngine([rule])
    modified: bool = engine.apply_passes(graph)  # type: ignore[arg-type]
    assert modified
    assert graph.outputs == ["fused"]


def test_dce_subgraph_side_effects_and_nested_prune() -> None:
    """Test DCE pass handling of subgraphs with side effects and non-IRGraph attributes."""
    subgraph_with_print: LogicalGraph = LogicalGraph(name="sub_print")
    subgraph_with_print.nodes = {
        "p1": LogicalNode(id="p1", op_type="Print", inputs=[]),
    }
    container_node: LogicalNode = LogicalNode(
        id="c1",
        op_type="Cond",
        inputs=[],
        subgraphs={"then_branch": subgraph_with_print, "not_a_graph": "invalid"},  # type: ignore[arg-type]
    )
    assert _node_has_side_effects(container_node)

    # Subgraph in attributes with side effect (hits line 49)
    attr_node = LogicalNode(
        id="c_attr",
        op_type="Custom",
        inputs=[],
        attributes={"subgraph": subgraph_with_print},
    )
    assert _node_has_side_effects(attr_node)

    # Pure subgraph in subgraphs dict (hits fallthrough without return True)
    subgraph_no_mod: LogicalGraph = LogicalGraph(name="sub_no_mod")
    subgraph_no_mod.nodes = {
        "out": LogicalNode(id="out", op_type="Add", inputs=[]),
    }
    subgraph_no_mod.outputs = ["out"]
    pure_container = LogicalNode(
        id="c_pure",
        op_type="WhileLoop",
        inputs=[],
        subgraphs={"body": subgraph_no_mod, "not_graph": 42},  # type: ignore[arg-type]
        attributes={"subgraph": subgraph_no_mod},
    )
    assert not _node_has_side_effects(pure_container)

    # _prune_nested_subgraphs where sub_mod is False in attributes (hits branch 84 -> 80)
    prune_res: bool = _prune_nested_subgraphs(pure_container)
    assert not prune_res


def test_llvm_cpp_validate_mlir_import_error_and_visit_in0(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test validate_mlir_operation on ImportError and edge handling in CppGenerator."""
    with monkeypatch.context() as m:
        m.setattr(cpp_gen, "get_supported_mlir_dialects", lambda: (_ for _ in ()).throw(ImportError("mock")))
        assert not cpp_gen.validate_mlir_operation("test.op")

    graph: LogicalGraph = LogicalGraph(name="test_cpp_edge")
    prod: LogicalNode = LogicalNode(id="prod", op_type="Split", inputs=[], outputs=["out0", "out1"])
    n: LogicalNode = LogicalNode(id="add_node", op_type="Add", inputs=["out1", "out1"])
    graph.nodes = {"prod": prod, "add_node": n}

    gen: cpp_gen.CppGenerator = cpp_gen.CppGenerator(graph=graph)
    gen._visit_node(n, graph)
    assert any("out1" in line for line in gen.lines)


def test_device_clear_cache_with_backend_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clear_cache when active backend implements clear_cache."""
    from ml_switcheroo_compiler.core.device import clear_cache

    class BackendWithCache:
        """Mock backend supporting clear_cache."""

        called: bool = False

        @classmethod
        def clear_cache(cls) -> None:
            """Record cache clearance call."""
            cls.called = True

    with monkeypatch.context() as m:
        m.setattr("ml_switcheroo_compiler.backends.registry.get_active_backend", lambda: BackendWithCache)
        clear_cache()
        assert BackendWithCache.called


def test_diagnostics_debugging_dump_info(tmp_path: pathlib.Path) -> None:
    """Test enable_dump_debug_info creates directory."""
    from ml_switcheroo_compiler.diagnostics.debugging import enable_dump_debug_info

    dump_dir = tmp_path / "debug_dump"
    enable_dump_debug_info(str(dump_dir), tensor_debug_mode="FULL_TENSOR", circular_buffer_size=1000)
    assert dump_dir.exists()


def test_diagnostics_summary_encode_image_1channel() -> None:
    """Test encode_image handles 3D tensors with single trailing channel."""
    from ml_switcheroo_compiler.diagnostics.summary import encode_image

    arr = np.ones((8, 8, 1), dtype=np.float32)
    png_bytes = encode_image(arr)
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG")


def test_conv_ops_convolve_infer_shape_custom_mode() -> None:
    """Test Convolve.infer_shape with unmapped custom mode and invalid input."""
    from ml_switcheroo_compiler.ops.linalg.conv_ops import Convolve

    class DummyArray:
        """Dummy array container with shape."""

        shape = (10,)

    res = Convolve().infer_shape(DummyArray(), DummyArray(), mode="unknown")
    assert res == (19,)

    assert Convolve().infer_shape(None, None) == ()


def test_time_distributed_infer_shape_valid() -> None:
    """Test TimeDistributed.infer_shape when inner operation returns shape, rank < 2, and eager func."""
    from ml_switcheroo_compiler.ops.nn.time_distributed import TimeDistributed, time_distributed

    class DummyInput:
        """Dummy tensor with batch and feature dimensions."""

        shape = (2, 5, 8, 16)
        dtype = "float32"

    class FakeOp:
        """Mock inner operation returning shaped tensor."""

        def infer_shape(self, inp: DummyInput, **kwargs: object) -> tuple[int, ...]:
            """Infer output shape from input."""
            return (inp.shape[0], 64)

    class FakeOpEmpty:
        """Mock inner operation returning empty shape."""

        def infer_shape(self, inp: object, **kwargs: object) -> tuple[int, ...]:
            """Infer empty shape."""
            return ()

    td = TimeDistributed()
    with pytest.MonkeyPatch().context() as m:
        m.setattr("ml_switcheroo_compiler.ops.nn.time_distributed.get_op", lambda name: FakeOp)
        assert td.infer_shape(DummyInput(), wrapped_op_name="Fake") == (2, 5, 64)

        m.setattr("ml_switcheroo_compiler.ops.nn.time_distributed.get_op", lambda name: FakeOpEmpty)
        assert td.infer_shape(DummyInput(), wrapped_op_name="FakeEmpty") == (2, 5, 8, 16)

        m.setattr(
            "ml_switcheroo_compiler.ops.nn.time_distributed.get_op",
            lambda name: (_ for _ in ()).throw(RuntimeError("error")),
        )
        assert td.infer_shape(DummyInput(), wrapped_op_name="FakeErr") == (2, 5, 8, 16)

    class Dummy1D:
        """Dummy 1D input."""

        shape = (5,)

    assert TimeDistributed().infer_shape(Dummy1D()) == (5,)
    assert TimeDistributed().infer_shape(DummyInput()) == (2, 5, 8, 16)

    t_in = Tensor(np.ones((2, 3, 4)), TensorConfig((2, 3, 4), "float32", "cpu"))
    with ConfigContext(eager_mode=False):
        global_tracing_state.start_tracing(LogicalGraph())
        try:
            res_traced = time_distributed(t_in, wrapped_op_name="Dense")
            assert res_traced is not None
        finally:
            global_tracing_state.stop_tracing()


def test_shape_joining_concatenate_rank0() -> None:
    """Test concatenate on rank-0 tensors outside eager mode."""
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.shape.joining import concatenate

    with ConfigContext(eager_mode=False):
        t1 = Tensor(np.array(1.0), TensorConfig((), "float32", "cpu"))
        t2 = Tensor(np.array(2.0), TensorConfig((), "float32", "cpu"))
        res = concatenate([t1, t2], axis=0)
        assert res.shape == ()


def test_shape_splitting_unstack_rank0() -> None:
    """Test unstack on rank-0 tensor outside eager mode."""
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.shape.splitting import unstack

    with ConfigContext(eager_mode=False):
        t = Tensor(np.array(42.0), TensorConfig((), "float32", "cpu"))
        res = unstack(t, axis=0)
        assert len(res) == 1
        assert res[0].shape == ()


def test_shape_slicing_branches() -> None:
    """Test slice outside eager mode with negative step, negative end, and out of bounds axis."""
    from ml_switcheroo_compiler.core.config import ConfigContext
    from ml_switcheroo_compiler.ops.shape.slicing import slice as slice_op

    with ConfigContext(eager_mode=False):
        t = Tensor(np.ones((10, 20)), TensorConfig((10, 20), "float32", "cpu"))
        # Negative step (s < 0)
        res_rev = slice_op(t, axis=0, start=8, end=2, step=-2)
        assert res_rev.shape == (3, 20)

        # Negative end (en < 0 and end is not None)
        res_neg_end = slice_op(t, axis=1, start=2, end=-4, step=1)
        assert res_neg_end.shape == (10, 14)

        # Positive end (en >= 0 and end is not None) -> hits branch 60 -> 63
        res_pos_end = slice_op(t, axis=1, start=2, end=5, step=1)
        assert res_pos_end.shape == (10, 3)

        # None end (end is None)
        res_none_end = slice_op(t, axis=1, start=2, end=None, step=1)
        assert res_none_end.shape == (10, 18)

        # Out of bounds axis (norm_axis >= rank)
        res_oob = slice_op(t, axis=5, start=0, end=5, step=1)
        assert res_oob.shape == (10, 20)


def test_backend_types_import_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ImportError branches and normal operation in backend types modules."""
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.dask.types as dask_types
    import ml_switcheroo_compiler.backends.keras.types as keras_types
    import ml_switcheroo_compiler.backends.mlx.types as mlx_types

    # Normal types functions
    assert dask_types.zeros(type, (2, 2)) is not None
    assert dask_types.array(type, [1, 2], dtype="float32") is not None
    assert dask_types.array(type, [1, 2]) is not None
    assert dask_types.asarray(type, [1, 2]) is not None
    assert np.isclose(dask_types.item(type, np.array(5.0)), 5.0)

    assert keras_types.zeros(type, (2, 2)) is not None
    assert keras_types.array(type, [1, 2], dtype="float32") is not None
    assert keras_types.array(type, [1, 2]) is not None
    assert keras_types.asarray(type, [1, 2]) is not None
    assert np.isclose(keras_types.item(type, np.array(5.0)), 5.0)

    assert mlx_types.zeros(type, (2, 2)) is not None
    assert mlx_types.array(type, [1, 2], dtype="float32") is not None
    assert mlx_types.array(type, [1, 2]) is not None
    assert mlx_types.asarray(type, [1, 2]) is not None
    assert np.isclose(mlx_types.item(type, np.array(5.0)), 5.0)

    # Keras asarray branches
    mock_kops_asarray = type("MockKops", (), {"asarray": staticmethod(lambda d: "asarray_res")})()
    with monkeypatch.context() as m:
        m.setattr(keras_types, "kops", mock_kops_asarray)
        assert keras_types.asarray(type, [1, 2]) == "asarray_res"

    mock_kops_none = type("MockKops", (), {"array": staticmethod(lambda d, dtype=None: "array_res")})()
    with monkeypatch.context() as m:
        m.setattr(keras_types, "kops", mock_kops_none)
        assert keras_types.asarray(type, [1, 2]) == "array_res"

    # MLX asarray branch
    mock_mx_asarray = type("MockMX", (), {"asarray": staticmethod(lambda d: "mx_asarray_res")})()
    with monkeypatch.context() as m:
        m.setattr(mlx_types, "mx", mock_mx_asarray)
        assert mlx_types.asarray(type, [1, 2]) == "mx_asarray_res"

    # Dask types fallback
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "dask", None)
        m.setitem(sys.modules, "dask.array", None)
        importlib.reload(dask_types)
        assert dask_types.da is None

    # Keras types fallback
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "keras", None)
        m.setitem(sys.modules, "keras.ops", None)
        importlib.reload(keras_types)
        assert keras_types.kops is None

    # MLX types fallback
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mlx", None)
        m.setitem(sys.modules, "mlx.core", None)
        importlib.reload(mlx_types)
        assert mlx_types.mx is None

    # Reload originals cleanly
    importlib.reload(dask_types)
    importlib.reload(keras_types)
    importlib.reload(mlx_types)


def test_pyarrow_compute_types_and_eager_coverage() -> None:
    """Test pyarrow_compute to_table default names, unsupported fallback, and eager error handling."""
    import pyarrow as pa

    import ml_switcheroo_compiler.backends.pyarrow_compute as pa_pkg
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    # 1. to_table without names (default names branch)
    c1 = pa.array([1, 2])
    tbl = pa_pkg.to_table([c1])
    assert tbl.column_names == ["f0"]

    # 2. to_table with unsupported type (falls through to return actual_data)
    assert pa_pkg.to_table("unsupported_string") == "unsupported_string"

    # 3. eager.execute_op when pc_mod direct fn raises exception
    with pytest.raises(BackendNotSupportedError, match="Failed executing pyarrow.compute"):
        pa_pkg.execute_op("Add", "not_an_arrow_array", "invalid")

    # 4. eager.execute_op when pc_mod mapped fn (e.g. Where -> if_else) raises exception (lines 194-196)
    with pytest.raises(BackendNotSupportedError, match="Failed executing pyarrow.compute.if_else"):
        pa_pkg.execute_op("Where", "invalid", "inputs")


def test_awkward_eager_and_types_full_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test full coverage of remaining awkward eager and types fallback branches.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.
    """
    import importlib
    import sys
    from unittest.mock import MagicMock, patch

    import ml_switcheroo_compiler.backends.awkward.eager as ak_eager
    import ml_switcheroo_compiler.backends.awkward.types as ak_types
    from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig

    # 1. eager Tensor unwrap (lines 20-21)
    t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), "float32", "cpu"))
    assert np.allclose(ak_eager.execute_op("Add", t, t), [2.0, 4.0])

    # 2. eager ak_mod is None fallback (lines 143-144, branch 149->158)
    with patch.dict(sys.modules, {"awkward": None}):
        assert np.allclose(ak_eager.execute_op("Add", np.array([1.0]), np.array([2.0])), [3.0])

    # 3. types _get_ak_module fallback (lines 17-18)
    orig_import = importlib.import_module

    def mock_import(name: str) -> object:
        if name == "awkward":
            raise ImportError("no awkward")
        return orig_import(name)

    with patch("importlib.import_module", side_effect=mock_import):
        mod = ak_types._get_ak_module()
        assert getattr(mod, "__name__", "") == "numpy"

    # 4. types zeros fallback without Array (line 37, branch 35->37)
    mock_ak_no_arr = MagicMock(spec=[])
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        z = ak_types.zeros((2, 2))
        assert isinstance(z, np.ndarray)

    # 5. types array and ragged_array without Array and with ValueError (lines 62-63, 113-114, branch 54->59)
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        with patch("numpy.array", side_effect=[ValueError("ragged"), np.array([1], dtype=object)]):
            a = ak_types.array([[1], [2, 3]])
            assert a.dtype == object

        with patch("numpy.array", side_effect=[ValueError("ragged"), np.array([1], dtype=object)]):
            ra = ak_types.ragged_array([[1], [2, 3]])
            assert ra.dtype == object

    # 6. types asarray fallback (lines 81-84, branch 78->83)
    mock_ak_err = MagicMock()
    mock_ak_err.Array.side_effect = RuntimeError("fail")
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_err):
        as_arr = ak_types.asarray([1, 2])
        assert isinstance(as_arr, np.ndarray)

    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        as_arr2 = ak_types.asarray([1, 2])
        assert isinstance(as_arr2, np.ndarray)

    # 7. types record_array return actual_data (line 137, branch 136->137)
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        assert ak_types.record_array("non_mapping_str") == "non_mapping_str"
    # types record_array with non-mapping and ak_mod with Array (line 137)
    assert ak_types.record_array([1, 2]) is not None

    # 8. types with_field return actual_base (line 223, branch 219->223)
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        assert ak_types.with_field("non_dict_str", where="x") == "non_dict_str"

    # 9. types unzip return (actual_data,) (line 246, branch 244->246)
    with patch.object(ak_types, "_get_ak_module", return_value=mock_ak_no_arr):
        assert ak_types.unzip("non_dict_str") == ("non_dict_str",)

    # 10. types item with actual_data.item() (line 268) and 1D list from to_list (line 265)
    class MockScalar:
        """Mock scalar container with item method."""

        def item(self) -> float:
            return 42.0

    assert ak_types.item(MockScalar()) == 42.0

    class Mock1DList:
        """Mock container returning 1D list from to_list."""

        def to_list(self) -> list[float]:
            return [9.5]

    assert ak_types.item(Mock1DList()) == 9.5

    class Mock2DList:
        """Mock container returning 2D list from to_list."""

        def to_list(self) -> list[list[float]]:
            return [[14.5]]

    assert ak_types.item(Mock2DList()) == 14.5
