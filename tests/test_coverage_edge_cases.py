"""Exhaustive tests targeting edge cases and branch coverage across the compiler."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
    import torch.distributed as dist
except Exception:
    torch = None
    dist = None

from ml_switcheroo_compiler.backends.dask.profiler import DaskProfiler, _sync_dask_result
from ml_switcheroo_compiler.backends.edge.config_models import load_webrtc_collectives
from ml_switcheroo_compiler.backends.edge.profiler import EdgeProfiler
from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import get_wasm_simd_op, get_wasm_simd_ops
from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
    _load_templates,
    get_wgsl_kernels_config,
    get_wgsl_op_mapping,
)
from ml_switcheroo_compiler.backends.keras.profiler import KerasProfiler, _get_keras_peak_memory_mb, _sync_keras_result
from ml_switcheroo_compiler.backends.numba.profiler import NumbaProfiler
from ml_switcheroo_compiler.backends.sparse.profiler import SparseProfiler
from ml_switcheroo_compiler.backends.tensorflow.profiler import (
    TensorFlowProfiler,
    _get_tf_peak_memory_mb,
    _sync_tf_result,
)
from ml_switcheroo_compiler.benchmarks.orchestrator import BenchmarkOrchestrator
from ml_switcheroo_compiler.core.type_promotion import _load_declarative_promotion_lattice
from ml_switcheroo_compiler.grad.checkpointing import load_rematerialization_rules
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode
from ml_switcheroo_compiler.ops.registry import get_all_ops
from ml_switcheroo_compiler.ops.shape_inference import infer_shape
from ml_switcheroo_compiler.tracing.builder import TracingNodeBuilder
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import _load_autodiff_rule
from ml_switcheroo_compiler.transforms.autodiff_rules.custom_rules import (
    _cond_jvp,
    _cond_vjp,
    _scan_vjp,
    _while_loop_jvp,
    _while_loop_vjp,
)
from ml_switcheroo_compiler.transforms.passes.config_models import load_mesh_partitioning
from ml_switcheroo_compiler.transforms.passes.constant_folding import _evaluate_constant_node
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass


def test_shape_inference_pass_attribute_error_and_none_branches() -> None:
    """Test shape_inference_pass with unhandled op (None shape) and node raising AttributeError on setting strides.

    Returns:
        None
    """

    class NodeWithAttributeErrorStrides(LogicalNode):
        """Node whose strides attribute raises AttributeError on set."""

        @property
        def strides(self) -> tuple[int, ...]:
            """Return empty strides.

            Returns:
                tuple[int, ...]: Strides tuple.
            """
            return ()

        @strides.setter
        def strides(self, value: tuple[int, ...]) -> None:
            """Raise AttributeError on set.

            Args:
                value (tuple[int, ...]): New strides.

            Raises:
                AttributeError: Always raised.
            """
            raise AttributeError("Read only strides")

    node_none = LogicalNode(id="n_none", op_type="UnknownOpForNoneShape", inputs=[], attributes={}, shape_metadata=None)
    node_custom = NodeWithAttributeErrorStrides(id="n_custom", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 3))
    graph = IRGraph(name="test_attr_err", nodes={"n_none": node_none, "n_custom": node_custom}, outputs=["n_custom"])

    with patch("ml_switcheroo_compiler.ops.shape_inference.infer_shape", side_effect=[None, (2, 3)]):
        modified = shape_inference_pass(graph)
        assert isinstance(modified, bool)


def test_dce_outputs_branches() -> None:
    """Test DCE pass when outputs are unchanged and when dead outputs are pruned.

    Returns:
        None
    """
    node = LogicalNode(id="in_0", op_type="Input")
    # 1. Unmodified outputs
    graph1 = IRGraph(name="dce_unmodified", nodes={"in_0": node}, outputs=["in_0"])
    res1 = dce_pass(graph1)
    assert res1 is False

    # 2. Dead output pruned
    graph2 = IRGraph(name="dce_modified", nodes={"in_0": node}, outputs=["in_0", "dead_out"])
    res2 = dce_pass(graph2)
    assert res2 is True
    assert graph2.outputs == ["in_0"]


def test_constant_folding_backend_item_branch() -> None:
    """Test constant folding pass with backend having item method and direct _evaluate_constant_node.

    Returns:
        None
    """

    class MockBackendWithItem:
        """Mock backend with item method."""

        def item(self, val: object) -> float:
            """Return float item.

            Args:
                val (object): Input value.

            Returns:
                float: Unwrapped item.
            """
            return 42.0

    node = LogicalNode(id="c0", op_type="Add", inputs=[], attributes={}, shape_metadata=(1,))
    graph = IRGraph(name="cf_backend_item", nodes={"c0": node})
    with patch("ml_switcheroo_compiler.transforms.passes.constant_folding.evaluate_graph", return_value={"c0": np.array([42.0])}):
        val = _evaluate_constant_node(node, [], graph, MockBackendWithItem())
        assert val == 42.0


def test_device_mesh_partitioning_config_custom_path() -> None:
    """Test load_mesh_partitioning with explicit path argument.

    Returns:
        None
    """
    import ml_switcheroo_compiler.transforms.passes.config_models as cm

    real_path = os.path.join(os.path.dirname(cm.__file__), "spmd_mappings", "mesh_partitioning.yaml")
    cfg = load_mesh_partitioning(path=real_path)
    assert cfg is not None


def test_custom_vjp_rules_empty_attributes() -> None:
    """Test _cond_vjp, _scan_vjp, _while_loop_vjp, _cond_jvp, and _while_loop_jvp with empty attributes.

    Returns:
        None
    """
    graph = IRGraph(name="vjp_test")
    node = LogicalNode(id="flow_op", op_type="Cond", inputs=["in_cond", "in_data"], attributes={})

    cond_res = _cond_vjp(graph, node, "ct")
    assert len(cond_res) == 2

    scan_res = _scan_vjp(graph, node, "ct")
    assert len(scan_res) == 2

    while_vjp_res = _while_loop_vjp(graph, node, "ct")
    assert len(while_vjp_res) == 2

    while_jvp_res = _while_loop_jvp(graph, node, "t")
    assert while_jvp_res is not None

    cond_jvp_res = _cond_jvp(graph, node, ["t1", "t2"])
    assert cond_jvp_res == "mock_tangent"


def test_autodiff_provider_missing_rule_type() -> None:
    """Test _load_autodiff_rule with non-existent rule type.

    Returns:
        None
    """
    rule = _load_autodiff_rule("CustomOpNoRules", "custom_nonexistent")
    assert rule is None or rule == {}


def test_builder_infer_shape_none_branch() -> None:
    """Test TracingNodeBuilder.emit_tracing_node when infer_shape returns None.

    Returns:
        None
    """
    from ml_switcheroo_compiler.tracing import global_tracing_state

    global_tracing_state.start_tracing("test_infer_none")
    try:
        with patch("ml_switcheroo_compiler.tracing.builder.infer_shape", return_value=None):
            tensor = TracingNodeBuilder.emit_tracing_node("CustomOpWithNoneShape")
            assert tensor.shape == ()
    finally:
        global_tracing_state.stop_tracing()


def test_ops_registry_get_all_ops_key_error() -> None:
    """Test get_all_ops handling of KeyError when get_op fails.

    Returns:
        None
    """
    with patch("ml_switcheroo_compiler.ops.registry.get_op", side_effect=KeyError("Missing op")):
        ops = get_all_ops()
        assert isinstance(ops, dict)


def test_ops_shape_inference_inputs_type_and_missing_op() -> None:
    """Test infer_shape with non-sequence inputs kwarg and non-existent op.

    Returns:
        None
    """
    # inputs is not a list or tuple
    res1 = infer_shape("Add", inputs=12345)
    assert res1 == ()

    # non-existent op with get_op returning None
    with patch("ml_switcheroo_compiler.ops.registry.get_op", return_value=None):
        res2 = infer_shape("NonExistentOpXYZ_12345")
        assert res2 == ()


def test_rematerialization_rules_existing_target_path() -> None:
    """Test load_rematerialization_rules with explicit existing target_path.

    Returns:
        None
    """
    import ml_switcheroo_compiler.transforms.passes as p

    real_path = os.path.join(os.path.dirname(p.__file__), "rematerialization_rules.yaml")
    cfg = load_rematerialization_rules(path=real_path)
    assert cfg is not None


def test_type_promotion_lattice_exceptions_and_missing_file() -> None:
    """Test _load_declarative_promotion_lattice error branches.

    Returns:
        None
    """
    # Test ValueError on invalid outer dtype
    mock_yaml1 = {"lattice": {"InvalidDTypeKey": {"OtherInvalid": "InvalidResult"}}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_yaml1):
        _load_declarative_promotion_lattice()

    # Test ValueError on valid outer dtype but invalid inner dtype (hitting lines 192-193)
    mock_yaml2 = {"lattice": {"float32": {"InvalidInnerDType": "float32"}}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_yaml2):
        _load_declarative_promotion_lattice()

    # Test file missing
    with patch("os.path.exists", return_value=False):
        _load_declarative_promotion_lattice()


def test_benchmarks_orchestrator_invalid_profile_branch() -> None:
    """Test _load_manifest_profiles with invalid non-dict profiles entry.

    Returns:
        None
    """
    mock_manifest = {"profiles": {"invalid_backend": "not_a_dict_payload"}}
    with patch("pathlib.Path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_manifest):
        specs = BenchmarkOrchestrator._load_manifest_profiles()
        assert specs == {}


def test_edge_webrtc_collectives_config_explicit_path() -> None:
    """Test load_webrtc_collectives with explicit path.

    Returns:
        None
    """
    import ml_switcheroo_compiler.backends.edge.config_models as ecm

    real_path = os.path.join(os.path.dirname(ecm.__file__), "webrtc_collectives.yaml")
    cfg = load_webrtc_collectives(path=real_path)
    assert cfg is not None


def test_webgl_generator_conv2d_shape_branches() -> None:
    """Test WebGLCodeGenerator with 4D Conv2D and non-int symbolic dimensions.

    Returns:
        None
    """
    graph = IRGraph(name="webgl_conv")
    node_in = LogicalNode(id="in_0", op_type="Input", shape_metadata=(1, 3, 32, 32))
    node_w = LogicalNode(id="w_0", op_type="Input", shape_metadata=(16, 3, 3, 3))
    node_conv = LogicalNode(id="conv_0", op_type="Conv2D", inputs=["in_0", "w_0"], shape_metadata=(1, 16, 30, 30))
    graph.nodes = {"in_0": node_in, "w_0": node_w, "conv_0": node_conv}
    graph.inputs = ["in_0", "w_0"]
    graph.outputs = ["conv_0"]

    gen = WebGLCodeGenerator(graph)
    assert gen.generate() is not None

    # Test with non-int dimension (symbolic dimension string) hitting branch 71->70
    graph_sym = IRGraph(name="webgl_sym")
    node_sym = LogicalNode(id="sym_add", op_type="Add", inputs=[], shape_metadata=("batch", 4, 8))
    graph_sym.nodes = {"sym_add": node_sym}
    graph_sym.inputs = []
    graph_sym.outputs = ["sym_add"]
    gen_sym = WebGLCodeGenerator(graph_sym)
    assert gen_sym.generate() is not None


def test_webgpu_visit_broadcast_and_empty_outputs_exception() -> None:
    """Test WebGPU visit_Broadcast and compile_forward_backward empty outputs.

    Returns:
        None
    """
    graph = IRGraph(name="webgpu_extra")
    node_bc = LogicalNode(id="bc_0", op_type="Broadcast", inputs=["in_0"], shape_metadata=(4,))
    node_out = LogicalNode(id="out_0", op_type="Relu", inputs=["bc_0"], shape_metadata=(4,))
    graph.nodes = {"bc_0": node_bc, "out_0": node_out}
    graph.inputs = ["in_0"]
    graph.outputs = ["out_0"]

    gen = WebGPUCodeGenerator(graph)
    lines, x, y, z = gen.visit_Broadcast(node_bc, ["buf_in0_f32"])
    assert len(lines) > 0
    assert (x, y, z) == ("1", "1", "1")

    # 1. target_output=None with non-empty outputs (hitting branch 962->963)
    mock_bwd = IRGraph(name="mock_bwd", nodes={"out_0": node_out})
    mock_bwd.inputs = []
    with patch("ml_switcheroo_compiler.transforms.autodiff.grad", return_value=mock_bwd):
        step_code = gen.generate_training_step(target_output=None)
        assert len(step_code) > 0

    # 2. Empty outputs in generate_training_step raises ValueError
    graph_empty = IRGraph(name="empty_outputs", nodes={})
    graph_empty.inputs = []
    graph_empty.outputs = []
    gen_empty = WebGPUCodeGenerator(graph_empty)
    with pytest.raises(ValueError, match="Target output cannot be identified"):
        gen_empty.generate_training_step(target_output=None)


def test_wgsl_provider_kernels_config_and_global_bindings() -> None:
    """Test get_wgsl_kernels_config, get_wgsl_template_for_op, and _load_templates global_bindings merge branch.

    Returns:
        None
    """
    cfg = get_wgsl_kernels_config()
    assert isinstance(cfg, dict)

    # Test get_wgsl_op_mapping with non-dict op mapping hitting branch 47->49
    with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value={"op_mappings": {"invalid_op": "not_a_dict"}}):
        assert get_wgsl_op_mapping("invalid_op") == {}

    mock_base = {"templates": {}}
    mock_kernels = {"templates": {"mock_k": {}}, "bindings": {"global_bindings": "// global bindings test"}}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=mock_base):
        with patch("ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider._load_kernels", return_value=mock_kernels):
            import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

            wp._WGSL_TEMPLATES = {}
            _load_templates()
            assert wp._WGSL_TEMPLATES.get("global_bindings") == "// global bindings test"


def test_wasm_simd_ops_retrieval() -> None:
    """Test get_wasm_simd_ops and get_wasm_simd_op.

    Returns:
        None
    """
    ops = get_wasm_simd_ops()
    assert isinstance(ops, dict)
    add_op = get_wasm_simd_op("Add")
    assert isinstance(add_op, dict)
    missing_op = get_wasm_simd_op("NonExistentSimdOp")
    assert missing_op == {}

    # Test with non-dict op value hitting branches 96->101 and 116->118
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_simd_ops", return_value={"operations": {"bad_op": "not_a_dict"}}):
        assert get_wasm_simd_op("bad_op") == {}


def test_edge_profiler_prepare_inputs_fallback_and_empty_step() -> None:
    """Test EdgeProfiler._prepare_inputs key mismatch and empty inputs step.

    Returns:
        None
    """
    profiler = EdgeProfiler()
    graph = IRGraph(name="edge_test")
    node = LogicalNode(id="in_expected", op_type="Input")
    graph.nodes = {"in_expected": node}

    # Keys mismatch
    res_inputs = profiler._prepare_inputs(graph, {"in_other": [1.0, 2.0]})
    assert res_inputs == [[1.0, 2.0]]

    # Empty inputs execution step
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert res_empty["latency_ms"] >= 0.0


def test_numba_profiler_branches() -> None:
    """Test NumbaProfiler _prepare_inputs, _compile_graph, and profile_graph branches.

    Returns:
        None
    """
    profiler = NumbaProfiler()
    graph = IRGraph(name="numba_test")
    node_in = LogicalNode(id="in_0", op_type="Input")
    node_op = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    graph.nodes = {"in_0": node_in, "add_0": node_op}
    graph.inputs = ["in_0"]

    # 1. ndarray input with matching keys
    arr = np.array([1.0, 2.0], dtype=np.float32)
    inputs1 = {"in_0": arr}
    assert len(profiler._prepare_inputs(graph, inputs1)) == 1

    # 2. Mismatched keys fallback
    inputs2 = {"other_key": arr}
    assert len(profiler._prepare_inputs(graph, inputs2)) == 1

    # 3. Direct _compile_graph execution hitting lines 70-76
    mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
    with patch.dict("sys.modules", {"numba": MagicMock()}):
        with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value=mock_code):
            fn = profiler._compile_graph(graph)
            assert callable(fn)
            # profile_graph with genuine compiled function hitting lines 107-108
            res_compiled = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
            assert len(res_compiled["latencies"]) == 2

        # 4. _compile_graph with invalid code (no evaluate function)
        with patch("ml_switcheroo_compiler.backends.numba.generator.NumbaGenerator.generate", return_value="x = 1\n"):
            fn_invalid = profiler._compile_graph(graph)
            assert fn_invalid is None

        # 5. profile_graph with empty inputs hitting line 120
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert len(res_empty["latencies"]) == 2


def test_sparse_profiler_branches() -> None:
    """Test SparseProfiler branches: COO input, ndarray input, mismatched keys, and compiled execution.

    Returns:
        None
    """
    profiler = SparseProfiler()
    graph = IRGraph(name="sparse_test")
    node_in1 = LogicalNode(id="in_0", op_type="Input")
    node_in2 = LogicalNode(id="in_1", op_type="Input")
    node_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"])
    graph.nodes = {"in_0": node_in1, "in_1": node_in2, "add_0": node_add}
    graph.inputs = ["in_0", "in_1"]

    arr1 = np.array([1.0, 2.0], dtype=np.float32)
    arr2 = np.array([3.0, 4.0], dtype=np.float32)

    class MockCOO:
        """Mock COO array."""

        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def todense(self) -> np.ndarray:
            return self.arr

        @classmethod
        def from_numpy(cls, arr: np.ndarray) -> "MockCOO":
            return cls(arr)

    mock_sparse_module = MagicMock()
    mock_sparse_module.COO = MockCOO
    mock_sparse_module.GCXS = MockCOO

    with patch("ml_switcheroo_compiler.backends.sparse.profiler.sparse", mock_sparse_module):
        # 1. _prepare_inputs with ndarray and COO
        inputs1 = {"in_0": arr1, "in_1": MockCOO(arr2)}
        prep1 = profiler._prepare_inputs(graph, inputs1)
        assert len(prep1) == 2

        # 2. _prepare_inputs with mismatched keys
        inputs2 = {"other_0": arr1, "other_1": arr2}
        prep2 = profiler._prepare_inputs(graph, inputs2)
        assert len(prep2) == 2

        # 3. Direct _compile_graph execution hitting lines 73-79
        mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
        with patch.dict("sys.modules", {"sparse": mock_sparse_module}):
            with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value=mock_code):
                fn = profiler._compile_graph(graph)
                assert callable(fn)
                # profile_graph with genuine compiled function hitting lines 110-111
                res_compiled = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
                assert len(res_compiled["latencies"]) == 2

        # 4. _compile_graph with invalid code (no evaluate function)
        with patch("ml_switcheroo_compiler.backends.sparse.generator.SparseGenerator.generate", return_value="x = 1\n"):
            fn_invalid = profiler._compile_graph(graph)
            assert fn_invalid is None

        # 5. profile_graph with empty inputs
        res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
        assert len(res_empty["latencies"]) == 2

        # 6. profile_graph fallback with multiple inputs
        with patch.object(profiler, "_compile_graph", return_value=None):
            res_multi = profiler.profile_graph(graph, inputs1, num_iters=2, warmup_iters=1)
            assert len(res_multi["latencies"]) == 2

            # Test branch when accum has no __add__
            class SimpleCOO(MockCOO):
                """COO class without __add__."""

            res_no_add = profiler.profile_graph(graph, {"in_0": SimpleCOO(arr1), "in_1": SimpleCOO(arr2)}, num_iters=1, warmup_iters=1)
            assert len(res_no_add["latencies"]) == 1


def test_dask_profiler_branches() -> None:
    """Test DaskProfiler branches: da.Array input, mismatched keys, _compile_graph, and compiled execution.

    Returns:
        None
    """
    profiler = DaskProfiler()
    graph = IRGraph(name="dask_test")
    node_in1 = LogicalNode(id="in_0", op_type="Input")
    node_in2 = LogicalNode(id="in_1", op_type="Input")
    node_add = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"])
    graph.nodes = {"in_0": node_in1, "in_1": node_in2, "add_0": node_add}
    graph.inputs = ["in_0", "in_1"]

    arr = np.array([1.0, 2.0], dtype=np.float32)

    # 1. _prepare_inputs with mismatched keys and compute attribute
    mock_compute_val = MagicMock()
    mock_compute_val.compute.return_value = arr
    inputs_mismatch = {"other_key": mock_compute_val}
    prep_mismatch = profiler._prepare_inputs(graph, inputs_mismatch)
    assert len(prep_mismatch) == 1

    # 2. _compile_graph valid function hitting lines 104-110
    mock_code = "def evaluate(inputs):\n    return inputs[0]\n"
    with patch.dict("sys.modules", {"dask": MagicMock(), "dask.array": MagicMock()}):
        with patch("ml_switcheroo_compiler.backends.dask.generator.DaskGenerator.generate", return_value=mock_code):
            fn = profiler._compile_graph(graph)
            assert callable(fn)
            # profile_graph with compiled function hitting lines 149-150
            res_comp = profiler.profile_graph(graph, {"in_0": arr, "in_1": arr}, num_iters=2, warmup_iters=1)
            assert len(res_comp["latencies"]) == 2

        # 3. _compile_graph invalid function raising TypeError on line 106
        with patch("ml_switcheroo_compiler.backends.dask.generator.DaskGenerator.generate", return_value="x = 1\n"):
            with pytest.raises(TypeError, match="Failed to generate valid evaluate function"):
                profiler._compile_graph(graph)

    # 4. _sync_dask_result with mock da.Array having compute()
    mock_da = MagicMock()
    mock_da.compute.return_value = arr
    _sync_dask_result(mock_da)
    mock_da.compute.assert_called_once()

    # 5. _sync_dask_result with compute error fallback
    mock_err_da = MagicMock()
    mock_err_da.compute.side_effect = RuntimeError("compute fail")
    _sync_dask_result(mock_err_da)

    # 6. profile_graph with empty inputs
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    # 7. profile_graph fallback with multiple inputs hitting line 167
    with patch.object(profiler, "_compile_graph", return_value=None):
        res_fb = profiler.profile_graph(graph, {"in_0": arr, "in_1": arr}, num_iters=2, warmup_iters=1)
        assert len(res_fb["latencies"]) == 2


def test_keras_profiler_additional_branches() -> None:
    """Test KerasProfiler mismatched keys, _compile_graph, empty inputs, and memory/sync branches.

    Returns:
        None
    """
    profiler = KerasProfiler()
    graph = IRGraph(name="keras_test")
    node = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": node}
    graph.inputs = ["in_0"]

    arr = np.array([1.0, 2.0], dtype=np.float32)

    # 1. _prepare_inputs with mismatched keys
    prep = profiler._prepare_inputs(graph, {"other_key": arr})
    assert len(prep) == 1

    # 2. _compile_model valid function and profile_graph
    mock_code = "def get_model():\n    return lambda inputs: inputs[0]\n"
    with patch("ml_switcheroo_compiler.backends.keras.generator.KerasCodeGenerator.generate", return_value=mock_code):
        fn = profiler._compile_model(graph)
        assert callable(fn)
        res_comp = profiler.profile_graph(graph, {"in_0": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp["latencies"]) == 2

    # 3. _compile_model invalid function
    with patch("ml_switcheroo_compiler.backends.keras.generator.KerasCodeGenerator.generate", return_value="x = 1\n"):
        with pytest.raises(TypeError, match="Failed to generate valid get_model function"):
            profiler._compile_model(graph)

    # 4. profile_graph with empty inputs
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    # 5. _get_keras_peak_memory_mb without keras
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", None):
        assert _get_keras_peak_memory_mb() > 0.0

    # 6. _sync_keras_result with numpy() raising exception
    mock_val = MagicMock()
    mock_val.numpy.side_effect = RuntimeError("numpy fail")
    with patch("ml_switcheroo_compiler.backends.keras.profiler.keras", None):
        _sync_keras_result(mock_val)


def test_tensorflow_profiler_additional_branches() -> None:
    """Test TensorFlowProfiler mismatched keys, _compile_graph, empty inputs, and memory/sync branches.

    Returns:
        None
    """
    profiler = TensorFlowProfiler()
    graph = IRGraph(name="tf_test")
    node = LogicalNode(id="in_0", op_type="Input")
    graph.nodes = {"in_0": node}
    graph.inputs = ["in_0"]

    arr = np.array([1.0, 2.0], dtype=np.float32)

    # 1. _prepare_inputs with mismatched keys
    prep = profiler._prepare_inputs(graph, {"other_key": arr})
    assert len(prep) == 1

    # 2. _compile_graph valid function and profile_graph
    mock_code = "def apply_model(inputs):\n    return inputs[0]\n"
    with patch("ml_switcheroo_compiler.backends.tensorflow.generator.TensorFlowCodeGenerator.generate", return_value=mock_code):
        fn = profiler._compile_graph(graph)
        assert callable(fn)
        res_comp = profiler.profile_graph(graph, {"in_0": arr}, num_iters=2, warmup_iters=1)
        assert len(res_comp["latencies"]) == 2

    # 3. _compile_graph invalid function
    with patch("ml_switcheroo_compiler.backends.tensorflow.generator.TensorFlowCodeGenerator.generate", return_value="x = 1\n"):
        with pytest.raises(TypeError, match="Failed to generate valid apply_model function"):
            profiler._compile_graph(graph)

    # 4. profile_graph with empty inputs
    res_empty = profiler.profile_graph(IRGraph(), {}, num_iters=2, warmup_iters=1)
    assert len(res_empty["latencies"]) == 2

    # 5. _get_tf_peak_memory_mb without tf
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", None):
        assert _get_tf_peak_memory_mb() > 0.0

    # 6. _sync_tf_result with numpy() raising exception
    mock_val = MagicMock()
    mock_val.numpy.side_effect = RuntimeError("numpy fail")
    with patch("ml_switcheroo_compiler.backends.tensorflow.profiler.tf", None):
        _sync_tf_result(mock_val)


def test_profilers_compilation_failure_fallback() -> None:
    """Test all profilers falling back gracefully when compilation fails.

    Returns:
        None
    """
    graph = IRGraph(name="test_fail")
    node_in = LogicalNode(id="in_0", op_type="Input")
    node_op = LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])
    graph.nodes = {"in_0": node_in, "add_0": node_op}
    graph.inputs = ["in_0"]
    inputs = {"in_0": [1.0, 2.0]}

    # 1. NumbaProfiler
    p_numba = NumbaProfiler()
    with patch.object(p_numba, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_numba = p_numba.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_numba["latencies"]) == 1

    # 2. SparseProfiler
    p_sparse = SparseProfiler()
    with patch.object(p_sparse, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_sparse = p_sparse.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_sparse["latencies"]) == 1

    # 3. DaskProfiler
    p_dask = DaskProfiler()
    with patch.object(p_dask, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_dask = p_dask.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_dask["latencies"]) == 1

    # 4. KerasProfiler
    p_keras = KerasProfiler()
    with patch.object(p_keras, "_compile_model", side_effect=RuntimeError("comp fail")):
        res_keras = p_keras.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_keras["latencies"]) == 1

    # 5. TensorFlowProfiler
    p_tf = TensorFlowProfiler()
    with patch.object(p_tf, "_compile_graph", side_effect=RuntimeError("comp fail")):
        res_tf = p_tf.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res_tf["latencies"]) == 1


def test_backends_linker_edge_cases() -> None:
    """Test get_source_ast_ref edge cases in backends/linker.py.

    Returns:
        None
    """
    from ml_switcheroo_compiler.backends.linker import get_source_ast_ref

    # 1. frame is None
    with patch("inspect.currentframe", return_value=None):
        assert get_source_ast_ref() is None

    # 2. frame has no f_back (break loop)
    mock_frame = MagicMock()
    mock_frame.f_back = None
    with patch("inspect.currentframe", return_value=mock_frame):
        with patch("inspect.getframeinfo", return_value=MagicMock(filename="test.py", lineno=10)):
            ref = get_source_ast_ref(back_frames=5)
            assert ref == "test.py:10"

    # 3. inspect.getframeinfo raises exception
    with patch("inspect.getframeinfo", side_effect=ValueError("frame error")):
        assert get_source_ast_ref() is None


def test_backends_wgsl_empty_kernels() -> None:
    """Test get_wgsl_kernels_config with empty dict when file returns empty or non-dict.

    Returns:
        None
    """
    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    old_kernels = wp._WGSL_KERNELS
    try:
        wp._WGSL_KERNELS = {}
        with patch("builtins.open", MagicMock()):
            with patch("yaml.safe_load", return_value="not-a-dict"):
                conf = wp.get_wgsl_kernels_config()
                assert conf == {"bindings": {}, "op_mappings": {}, "templates": {}}
    finally:
        wp._WGSL_KERNELS = old_kernels


def test_backends_sparse_profiler_multi_input_accumulation() -> None:
    """Test SparseProfiler multi-input accumulation loop.

    Returns:
        None
    """
    p = SparseProfiler()
    graph = IRGraph(name="test_sparse_multi")
    graph.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "in_1": LogicalNode(id="in_1", op_type="Input")}
    graph.inputs = ["in_0", "in_1"]
    inputs = {"in_0": np.array([1.0, 2.0]), "in_1": np.array([3.0, 4.0])}
    with patch.object(p, "_compile_graph", return_value=None):
        res = p.profile_graph(graph, inputs, num_iters=1, warmup_iters=1)
        assert len(res["latencies"]) == 1


def test_backends_numpy_indexing_setitem() -> None:
    """Test _np_setitem in backends/numpy/eager/indexing.py.

    Returns:
        None
    """
    import ml_switcheroo_compiler.backends.numpy.eager.indexing  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    func = numpy_eager_registry.get("SetItem")
    assert func is not None
    x = np.array([10, 20, 30])
    res = func(np, x, 99, "1")
    assert res[1] == 99
    assert res[0] == 10


def test_backends_numpy_vision_power_iteration() -> None:
    """Test _np_power_iteration in backends/numpy/eager/vision_common.py.

    Returns:
        None
    """
    import ml_switcheroo_compiler.backends.numpy.eager.vision_common  # noqa: F401
    from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry

    func = numpy_eager_registry.get("PowerIteration")
    assert func is not None
    w = np.eye(3, dtype=np.float32)
    # Default u=None
    v, u, s = func(np, w, num_iters=2)
    assert v.shape == (3,)
    assert u.shape == (3,)

    # With u provided
    v2, u2, s2 = func(np, w, num_iters=1, u=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert v2.shape == (3,)


def test_backends_jax_distributed_collectives_branches() -> None:
    """Test jax_all_reduce branches (MAX, MIN, and import error).

    Returns:
        None
    """
    from ml_switcheroo_compiler.backends.jax.distributed_collectives import jax_all_reduce

    mock_jax = MagicMock()
    mock_jax.lax.psum.return_value = "sum_reduced"
    mock_jax.lax.pmax.return_value = "max_reduced"
    mock_jax.lax.pmin.return_value = "min_reduced"

    with patch.dict("sys.modules", {"jax": mock_jax, "jax.lax": mock_jax.lax}):
        assert jax_all_reduce("tensor", op="MAX", axis_name="x") == "max_reduced"
        assert jax_all_reduce("tensor", op="MIN", axis_name="x") == "min_reduced"

    with patch.dict("sys.modules", {"jax": None, "jax.lax": None}):
        assert jax_all_reduce("fallback_tensor", op="MAX", axis_name="x") == "fallback_tensor"


def test_backends_pytorch_distributed_collectives_branches() -> None:
    """Test pytorch distributed collectives when dist is initialized and for various ops.

    Returns:
        None
    """
    from ml_switcheroo_compiler.backends.pytorch.distributed_collectives import (
        pytorch_all_gather,
        pytorch_all_reduce,
        pytorch_broadcast,
        pytorch_reduce_scatter,
    )

    if torch is not None and dist is not None:
        t = torch.tensor([1.0, 2.0])

        with patch.object(dist, "is_available", return_value=True):
            with patch.object(dist, "is_initialized", return_value=True):
                with patch.object(dist, "all_reduce", return_value=None):
                    with patch.object(dist, "all_gather", return_value=None):
                        with patch.object(dist, "reduce_scatter", return_value=None):
                            with patch.object(dist, "broadcast", return_value=None):
                                with patch.object(dist, "get_world_size", return_value=2):
                                    # all_reduce branches
                                    res_prod = pytorch_all_reduce(t, op="PRODUCT")
                                    assert res_prod is not None
                                    res_max = pytorch_all_reduce(t, op="MAX")
                                    assert res_max is not None
                                    res_min = pytorch_all_reduce(t, op="MIN")
                                    assert res_min is not None
                                    res_sum = pytorch_all_reduce(t, op="SUM")
                                    assert res_sum is not None

                                    # all_gather
                                    res_gather = pytorch_all_gather(t, axis=0)
                                    assert res_gather is not None

                                    # reduce_scatter branches
                                    res_scatter_sum = pytorch_reduce_scatter(t, op="SUM", scatter_dim=0)
                                    assert res_scatter_sum is not None
                                    res_scatter_prod = pytorch_reduce_scatter(t, op="PROD", scatter_dim=0)
                                    assert res_scatter_prod is not None

                                    # broadcast
                                    res_bcast = pytorch_broadcast(t, src=0)
                                    assert res_bcast is not None

        # When dist not available/initialized
        with patch.object(dist, "is_available", return_value=False):
            assert pytorch_all_reduce("raw_tensor") == "raw_tensor"
            assert pytorch_all_gather("raw_tensor") == "raw_tensor"
            assert pytorch_reduce_scatter("raw_tensor") == "raw_tensor"
            assert pytorch_broadcast("raw_tensor") == "raw_tensor"

    # When torch or torch.distributed raises ImportError
    import builtins

    orig_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name in ("torch", "torch.distributed"):
            raise ImportError("torch missing")
        return orig_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=mock_import):
        assert pytorch_all_reduce("raw_tensor") == "raw_tensor"
        assert pytorch_all_gather("raw_tensor") == "raw_tensor"
        assert pytorch_reduce_scatter("raw_tensor") == "raw_tensor"
        assert pytorch_broadcast("raw_tensor") == "raw_tensor"


def test_backends_eager_group_ops_affine_branches() -> None:
    """Test _apply_affine_transform with weight and bias in core_group_ops.py.

    Returns:
        None
    """
    from ml_switcheroo_compiler.backends.eager.core_group_ops import _apply_affine_transform

    out = np.ones((2, 4), dtype=np.float32)
    weight = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    bias = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    res = _apply_affine_transform(np, out, axis=1, weight=weight, bias=bias)
    assert np.allclose(res, 3.0)


def test_backends_dask_and_tf_and_keras_profiler_edge_cases() -> None:
    """Test profiler edge cases for Dask, TensorFlow, and Keras.

    Returns:
        None
    """
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.dask as dask_pkg
    import ml_switcheroo_compiler.backends.dask.profiler as dp
    import ml_switcheroo_compiler.backends.keras as keras_pkg
    import ml_switcheroo_compiler.backends.keras.profiler as kp
    import ml_switcheroo_compiler.backends.tensorflow as tf_pkg
    import ml_switcheroo_compiler.backends.tensorflow.profiler as tp

    # 1. Module import fallbacks (lines 10-11)
    sys.modules["ml_switcheroo_compiler.backends.dask"] = dask_pkg
    with patch.dict(sys.modules, {"dask": None, "dask.array": None}):
        importlib.reload(dp)
        assert dp.da is None
    sys.modules["ml_switcheroo_compiler.backends.dask"] = dask_pkg
    importlib.reload(dp)

    sys.modules["ml_switcheroo_compiler.backends.keras"] = keras_pkg
    with patch.dict(sys.modules, {"keras": None}):
        importlib.reload(kp)
        assert kp.keras is None
    sys.modules["ml_switcheroo_compiler.backends.keras"] = keras_pkg
    importlib.reload(kp)

    sys.modules["ml_switcheroo_compiler.backends.tensorflow"] = tf_pkg
    with patch.dict(sys.modules, {"tensorflow": None}):
        importlib.reload(tp)
        assert tp.tf is None
    sys.modules["ml_switcheroo_compiler.backends.tensorflow"] = tf_pkg
    importlib.reload(tp)

    # 2. Dask sync computation exception and success
    mock_dask_lazy = MagicMock()
    mock_dask_lazy.compute.side_effect = RuntimeError("compute failed")
    assert dp._sync_dask_result(mock_dask_lazy) is mock_dask_lazy

    mock_dask_success = MagicMock()
    mock_dask_success.compute.return_value = 42
    assert dp._sync_dask_result(mock_dask_success) == 42
    assert dp._sync_dask_result([mock_dask_success]) == [42]
    assert dp._sync_dask_result((mock_dask_success,)) == (42,)
    assert dp._sync_dask_result({"res": mock_dask_success}) == {"res": 42}

    old_da = dp.da
    try:
        dp.da = None
        assert dp._sync_dask_result(mock_dask_success) is mock_dask_success
    finally:
        dp.da = old_da

    # 3. TensorFlowProfiler with device context and op nodes
    p_tf = TensorFlowProfiler()
    graph = IRGraph(name="test_tf_dev")
    graph.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input"), "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_0"])}
    graph.inputs = ["in_0"]
    inputs = {"in_0": [1.0, 2.0]}
    with patch.object(p_tf, "_compile_graph", return_value=lambda x: x):
        res_tf = p_tf.profile_graph(graph, inputs, num_iters=1, warmup_iters=1, device="/cpu:0")
        assert len(res_tf["latencies"]) == 1

    # 4. Keras peak memory with torch and tf mock modules and exceptions
    with patch("sys.platform", "linux"):
        mem_linux = kp._get_process_memory_mb()
        assert mem_linux > 0.0

    mock_torch_err = MagicMock()
    mock_torch_err.cuda.is_available.side_effect = RuntimeError("torch cuda fail")
    with patch.dict("sys.modules", {"torch": mock_torch_err}):
        _ = _get_keras_peak_memory_mb()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.max_memory_allocated.return_value = 10485760
    with patch.dict("sys.modules", {"torch": mock_torch}):
        mem = _get_keras_peak_memory_mb()
        assert mem > 0.0

    mock_torch_no_cuda = MagicMock()
    mock_torch_no_cuda.cuda.is_available.return_value = False
    mock_tf_err = MagicMock()
    mock_tf_err.config.experimental.get_memory_info.side_effect = RuntimeError("tf fail")
    with patch.dict("sys.modules", {"torch": mock_torch_no_cuda, "tensorflow": mock_tf_err}):
        _ = _get_keras_peak_memory_mb()

    mock_tf = MagicMock()
    mock_tf.config.experimental.get_memory_info.return_value = {"peak": 20971520}
    with patch.dict("sys.modules", {"torch": mock_torch_no_cuda, "tensorflow": mock_tf}):
        mem_tf = _get_keras_peak_memory_mb()
        assert mem_tf > 0.0

    # 5. _sync_keras_result with various nested types, convert_to_numpy, and exceptions
    _sync_keras_result({"a": [1.0, (2.0,)]})
    mock_result_err = MagicMock()
    mock_result_err.numpy.side_effect = RuntimeError("numpy fail")
    _sync_keras_result(mock_result_err)

    # Case A: no ops on keras
    mock_k_no_ops = MagicMock(spec=["backend"])
    mock_success = MagicMock()
    mock_success.numpy.return_value = 1.0
    mock_fail = MagicMock()
    mock_fail.numpy.side_effect = RuntimeError("numpy fail")
    with patch.object(kp, "keras", mock_k_no_ops):
        kp._sync_keras_result(mock_success)
        kp._sync_keras_result(mock_fail)

    # Case B: with ops on keras
    mock_k_ops = MagicMock()
    mock_k_ops.ops.convert_to_numpy.side_effect = [1.0, RuntimeError("ops fail")]
    with patch.object(kp, "keras", mock_k_ops):
        kp._sync_keras_result(mock_success)
        kp._sync_keras_result(mock_fail)

    # 6. KerasProfiler _prepare_inputs matching & mismatching keys
    p_k = KerasProfiler()
    g = IRGraph(name="test_inputs")
    g.nodes = {"in_0": LogicalNode(id="in_0", op_type="Input")}
    g.inputs = ["in_0"]
    r1 = p_k._prepare_inputs(g, {"in_0": [1.0]})
    assert len(r1) == 1
    r2 = p_k._prepare_inputs(g, {"mismatch": [1.0]})
    assert len(r2) == 1

    # 7. KerasProfiler _build_step_fn with ops.zeros and without ops
    mock_k_zeros = MagicMock()
    mock_k_zeros.ops.zeros.return_value = 0.0
    with patch.object(kp, "keras", mock_k_zeros):
        fn1 = p_k._build_step_fn(False, None, [])
        assert fn1() == 0.0

    with patch.object(kp, "keras", mock_k_no_ops):
        fn2 = p_k._build_step_fn(False, None, [])
        assert fn2() == 0.0

    # 8. KerasProfiler profile_graph branches
    graph_single = IRGraph(name="test_single")
    graph_single.nodes = {
        "in_0": LogicalNode(id="in_0", op_type="Input"),
        "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0"]),
    }
    graph_single.inputs = ["in_0"]
    with patch.object(p_k, "_compile_model", return_value=lambda x: x):
        res_single = p_k.profile_graph(graph_single, {"in_0": [1.0]}, num_iters=1, warmup_iters=1)
        assert len(res_single["latencies"]) == 1

    with patch.object(p_k, "_compile_model", side_effect=RuntimeError("compile fail")):
        res_comp_err = p_k.profile_graph(graph_single, {"in_0": [1.0]}, num_iters=1, warmup_iters=1)
        assert len(res_comp_err["latencies"]) == 1

    old_k = kp.keras
    try:
        kp.keras = None
        assert p_k._prepare_inputs(graph_single, {}) == []
        res_none = p_k.profile_graph(graph_single, {}, num_iters=1, warmup_iters=1)
        assert len(res_none["latencies"]) == 1
    finally:
        kp.keras = old_k

    # 9. KerasProfiler multi-input branch
    graph_multi = IRGraph(name="test_keras_multi")
    graph_multi.nodes = {
        "in_0": LogicalNode(id="in_0", op_type="Input"),
        "in_1": LogicalNode(id="in_1", op_type="Input"),
        "add_0": LogicalNode(id="add_0", op_type="Add", inputs=["in_0", "in_1"]),
    }
    graph_multi.inputs = ["in_0", "in_1"]
    inputs_multi = {"in_0": [1.0, 2.0], "in_1": [3.0, 4.0]}
    with patch.object(p_k, "_compile_model", return_value=lambda xs: xs[0]):
        res_multi = p_k.profile_graph(graph_multi, inputs_multi, num_iters=1, warmup_iters=1)
        assert len(res_multi["latencies"]) == 1
