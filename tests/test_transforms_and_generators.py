"""Behavioral tests for graph optimization passes, data-driven autodiff rules, and edge code generators."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from ml_switcheroo_ir import IRGraph, IRNode

from ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider import load_yaml_dir
from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider import (
    _load_templates,
    _load_webgpu_ops,
    get_js_orchestration_template,
    get_webgpu_ops,
    get_wgsl_global_bindings,
    get_wgsl_template,
)
from ml_switcheroo_compiler.ir.core import ensure_attrs
from ml_switcheroo_compiler.ops.generated_registry import OPS_REGISTRY
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import get_jvp_from_data, get_vjp_from_data
from ml_switcheroo_compiler.transforms.passes.control_flow_opt import (
    control_flow_optimization_pass,
    evaluate_static_predicate,
)
from ml_switcheroo_compiler.transforms.passes.parallel_scan import (
    _load_scan_rules,
    detect_associative_reduction,
    parallel_scan_pass,
)
from ml_switcheroo_compiler.transforms.passes.vectorization import _load_vmap_rules, vectorization_pass, vectorize_graph


def test_control_flow_opt_static_predicates_and_passes() -> None:
    """Verify `evaluate_static_predicate` and `control_flow_optimization_pass` across static/dynamic branches."""
    g = IRGraph()
    g.nodes["c_str"] = IRNode(id="c_str", op_type="Constant", attributes={"value": "unsupported_type"})
    assert evaluate_static_predicate(g, "c_str") is None

    g.nodes["add_node"] = IRNode(id="add_node", op_type="Add", inputs=["c_str"])
    assert evaluate_static_predicate(g, "add_node") is None

    g.nodes["in_var"] = IRNode(id="in_var", op_type="Input")
    g.nodes["c_num"] = IRNode(id="c_num", op_type="Constant", attributes={"value": 1})
    g.nodes["cmp_non_const"] = IRNode(id="cmp_non_const", op_type="Equal", inputs=["in_var", "c_num"])
    assert evaluate_static_predicate(g, "cmp_non_const") is None

    g.nodes["c_str2"] = IRNode(id="c_str2", op_type="Constant", attributes={"value": "other_str"})
    g.nodes["cmp_str"] = IRNode(id="cmp_str", op_type="Equal", inputs=["c_str", "c_str2"])
    assert evaluate_static_predicate(g, "cmp_str") is None

    g.nodes["cmp_single_in"] = IRNode(id="cmp_single_in", op_type="Equal", inputs=["c_num"])
    assert evaluate_static_predicate(g, "cmp_single_in") is None

    g.nodes["cmp_mixed"] = IRNode(id="cmp_mixed", op_type="Equal", inputs=["c_num", "c_str"])
    assert evaluate_static_predicate(g, "cmp_mixed") is None

    g.nodes["c_5"] = IRNode(id="c_5", op_type="Constant", attributes={"value": 5})
    g.nodes["c_3"] = IRNode(id="c_3", op_type="Constant", attributes={"value": 3})
    g.nodes["cmp_ge"] = IRNode(id="cmp_ge", op_type="GreaterEqual", inputs=["c_5", "c_3"])
    assert evaluate_static_predicate(g, "cmp_ge") is True

    g_opt = IRGraph()
    g_opt.nodes["pred_const"] = IRNode(id="pred_const", op_type="Constant", attributes={"value": True})
    f_block = IRGraph()
    f_block.nodes["fn"] = IRNode(id="fn", op_type="Identity", inputs=[])
    g_opt.nodes["cond_node"] = IRNode(
        id="cond_node",
        op_type="Cond",
        inputs=["pred_const"],
        attributes=ensure_attrs({"true_branch": None, "false_branch": f_block}),
    )
    res_graph = control_flow_optimization_pass(g_opt)
    assert isinstance(res_graph, IRGraph)
    assert "cond_node" in res_graph.nodes

    g_no_hoist = IRGraph()
    g_no_hoist.nodes["pred_dyn"] = IRNode(id="pred_dyn", op_type="Input")
    b1 = IRGraph()
    b1.nodes["b1_node"] = IRNode(id="b1_node", op_type="Add", inputs=["pred_dyn"])
    b2 = IRGraph()
    b2.nodes["b2_node"] = IRNode(id="b2_node", op_type="Multiply", inputs=["pred_dyn"])
    g_no_hoist.nodes["cond_dyn"] = IRNode(
        id="cond_dyn",
        op_type="Cond",
        inputs=["pred_dyn"],
        attributes=ensure_attrs({"true_branch": b1, "false_branch": b2}),
    )
    res_no_hoist = control_flow_optimization_pass(g_no_hoist)
    assert isinstance(res_no_hoist, IRGraph)
    assert "cond_dyn" in res_no_hoist.nodes


def test_parallel_scan_detection_and_pass_fallbacks() -> None:
    """Verify `parallel_scan_pass`, `detect_associative_reduction`, and malformed YAML scan rule fallback."""
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="['not_a_dict']")),
        patch("yaml.safe_load", return_value=["not_a_dict"]),
    ):
        assert _load_scan_rules() == {}

    b_graph = IRGraph()
    b_graph.nodes["unknown_op"] = IRNode(id="unknown_op", op_type="CustomUnsupportedOp")
    rules = {"associative_ops": {"Add": {"parallel_primitive": "CumSum"}}}
    assert detect_associative_reduction(b_graph, rules) is None

    g_scan = IRGraph()
    g_scan.nodes["scan_unassoc"] = IRNode(
        id="scan_unassoc",
        op_type="Scan",
        inputs=["init", "xs"],
        attributes=ensure_attrs({"body": b_graph}),
    )
    res_scan = parallel_scan_pass(g_scan)
    assert "scan_unassoc" in res_scan.nodes

    g_empty = IRGraph()
    g_empty.nodes["noop"] = IRNode(id="noop", op_type="Identity")
    assert parallel_scan_pass(g_empty) == g_empty


def test_vectorization_axis_alignment_and_policies() -> None:
    """Verify `vectorize_graph` and `vectorization_pass` across batch axis alignment and reshape policies."""
    with patch("os.path.exists", return_value=True), patch("yaml.safe_load", return_value={"no_rules_key": 1}):
        assert _load_vmap_rules() == {}

    g_align = IRGraph()
    g_align.inputs = ["ghost_input"]
    g_align.nodes["unmapped_input"] = IRNode(id="unmapped_input", op_type="Input")
    g_align.nodes["relu_op"] = IRNode(id="relu_op", op_type="Relu", inputs=["ghost_input"])
    g_align.outputs = ["unmapped_input", "relu_op"]
    v_align = vectorize_graph(g_align, in_axes=1, batch_size=2)
    assert "unmapped_input" in v_align.nodes
    assert "ghost_input_align_axis_0" in v_align.nodes

    g_shift = IRGraph()
    g_shift.inputs = ["x"]
    g_shift.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
    g_shift.nodes["red"] = IRNode(id="red", op_type="ReduceSum", inputs=["x"], attributes={"axis": -1, "dim": -1})
    g_shift.outputs = ["red"]
    v_shift = vectorize_graph(g_shift, in_axes=0, batch_size=2)
    assert "red" in v_shift.nodes

    g_reshape = IRGraph()
    g_reshape.inputs = ["x"]
    g_reshape.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
    g_reshape.nodes["resh1"] = IRNode(id="resh1", op_type="Reshape", inputs=["x"], attributes={"shape": (2, 2)})
    g_reshape.nodes["resh2"] = IRNode(id="resh2", op_type="Reshape", inputs=["x"], attributes={"newshape": (2, 2)})
    g_reshape.outputs = ["resh1", "resh2"]
    v_resh = vectorize_graph(g_reshape, in_axes=0, batch_size=2)
    assert "resh1" in v_resh.nodes and "resh2" in v_resh.nodes

    g_vmap_wrapper = IRGraph()
    g_vmap_wrapper.nodes["inp_short_shape"] = IRNode(id="inp_short_shape", op_type="Input", shape_metadata=(4,))
    g_vmap_wrapper.nodes["vmap_node"] = IRNode(
        id="vmap_node",
        op_type="Vmap",
        inputs=["inp_short_shape"],
        attributes=ensure_attrs({"in_axes": 5, "body": g_shift}),
    )
    g_vmap_wrapper.nodes["vmap_ghost"] = IRNode(
        id="vmap_ghost",
        op_type="Vmap",
        inputs=["ghost_not_in_graph"],
        attributes=ensure_attrs({"in_axes": 0, "body": g_shift}),
    )
    v_wrapper = vectorization_pass(g_vmap_wrapper)
    assert "vmap_node" in v_wrapper.nodes and "vmap_ghost" in v_wrapper.nodes


def test_autodiff_provider_public_vjp_and_jvp_rules() -> None:
    """Verify `get_vjp_from_data` and `get_jvp_from_data` expression parsing and finite-difference fallback."""
    custom_registry_entries = {
        "SyntheticComboOp": {
            "autodiff": {
                "vjp": [
                    "ZeroArgOp()",
                    "CustomAttrOp($input[0], foo=invalid_unquoted_literal)",
                    "BroadcastReduce($cotangent, $input[0])",
                    "BroadcastLike($cotangent, $input[1])",
                    "SetItem($input[0])",
                    "- $output",
                ]
            }
        },
        "SyntheticSqueezeAd": {"autodiff": {"vjp": ["ExpandDims($cotangent)"]}},
        "SyntheticExpandAd": {"autodiff": {"vjp": ["Squeeze($cotangent)"]}},
        "SyntheticReshapeAd": {"autodiff": {"vjp": ["Reshape($cotangent)"]}},
    }

    with patch.dict(OPS_REGISTRY, custom_registry_entries):
        g = IRGraph()
        g.nodes["a"] = IRNode(id="a", op_type="Input", shape_metadata=(2, 4))
        g.nodes["b"] = IRNode(id="b", op_type="Input", shape_metadata=(2, 4))
        g.nodes["cot"] = IRNode(id="cot", op_type="Input", shape_metadata=(2, 4))

        combo_vjp = get_vjp_from_data("SyntheticComboOp")
        assert combo_vjp is not None
        combo_node = IRNode(
            id="combo_n",
            op_type="DifferentOp",
            inputs=["a", "b"],
            attributes={"indexed": True},
            shape_metadata=(2, 4),
        )
        adjs = combo_vjp(g, combo_node, "cot")
        assert len(adjs) == 6
        assert g.nodes[adjs[1]].attributes.get("foo") == "invalid_unquoted_literal"
        assert g.nodes[adjs[2]].op_type == "BroadcastReduce"
        assert g.nodes[adjs[3]].op_type == "BroadcastLike"
        assert g.nodes[adjs[4]].op_type == "SetItem"
        assert g.nodes[adjs[5]].op_type == "Negative"

        sq_vjp = get_vjp_from_data("SyntheticSqueezeAd")
        assert sq_vjp is not None
        sq_out = sq_vjp(g, IRNode(id="sq_n", op_type="Squeeze", inputs=["a"], attributes={"axis": 1}), "cot")
        assert g.nodes[sq_out[0]].attributes.get("axis") == 1

        exp_vjp = get_vjp_from_data("SyntheticExpandAd")
        assert exp_vjp is not None
        exp_out = exp_vjp(g, IRNode(id="exp_n", op_type="ExpandDims", inputs=["a"], attributes={"axis": 1}), "cot")
        assert g.nodes[exp_out[0]].attributes.get("axis") == 1

        resh_vjp = get_vjp_from_data("SyntheticReshapeAd")
        assert resh_vjp is not None
        resh_out = resh_vjp(g, IRNode(id="r_n", op_type="Reshape", inputs=["a"]), "cot")
        assert g.nodes[resh_out[0]].shape_metadata == (2, 4)

    jvp_fn = get_jvp_from_data("UnregisteredFiniteDiffOp")
    assert jvp_fn is not None
    g_jvp = IRGraph()
    g_jvp.nodes["in1"] = IRNode(id="in1", op_type="Input", shape_metadata=(2,))
    g_jvp.nodes["in2"] = IRNode(id="in2", op_type="Input", shape_metadata=(2,))
    g_jvp.nodes["tan1"] = IRNode(id="tan1", op_type="Input", shape_metadata=(2,))
    jvp_id = jvp_fn(g_jvp, IRNode(id="n_multi", op_type="Add", inputs=["in1", "in2"]), ["tan1"])
    assert jvp_id in g_jvp.nodes

    with (
        patch("os.path.exists", side_effect=lambda p: str(p).endswith("LegacyFoundOp.yaml")),
        patch("builtins.open", mock_open(read_data="""LegacyFoundOp:\x0a  vjp: ['$cotangent']\x0a""")),
        patch("yaml.safe_load", return_value={"LegacyFoundOp": {"vjp": ["$cotangent"]}}),
    ):
        assert get_vjp_from_data("MissingLegacyOp") is None
        legacy_fn = get_vjp_from_data("LegacyFoundOp")
        assert legacy_fn is not None


def test_webgl_generate_with_empty_node_inputs() -> None:
    """Verify `WebGLCodeGenerator.generate()` handles `matmul` nodes with empty inputs using default `kDim`."""
    g = IRGraph(name="webgl_empty_inputs")
    g.nodes["test_node"] = IRNode(id="test_node", op_type="matmul", inputs=[], shape_metadata=(16, 16))
    g.outputs = ["test_node"]

    code = WebGLCodeGenerator(g).generate()
    assert "kDim" in code or "32" in code
    assert "prog_test_node" in code


def test_wasm_provider_yaml_dir_merging(tmp_path: Path) -> None:
    """Verify `load_yaml_dir` merges empty dicts and list values cleanly.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
    """
    yaml_dir = tmp_path / "yaml_test_dir"
    yaml_dir.mkdir()

    f1 = str(yaml_dir / "empty.yaml")
    with open(f1, "w", encoding="utf-8") as f:
        f.write("templates: {}")

    f2 = str(yaml_dir / "list_val.yaml")
    with open(f2, "w", encoding="utf-8") as f:
        f.write("""templates:
  raw_list:
    - item1
    - item2
""")

    with patch("pathlib.Path.is_dir", return_value=True), patch("glob.glob", return_value=[f1, f2]):
        res = load_yaml_dir("yaml_test_dir")
        assert "templates" in res
        assert "raw_list" in res["templates"]


def test_wgsl_provider_missing_and_malformed_files() -> None:
    """Verify `wgsl_provider` fallbacks when template or op YAML files are missing or non-dict."""
    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

    try:
        wp._WGSL_TEMPLATES = {}
        wp._WEBGPU_OPS = {}
        wp._WGSL_KERNELS = {}

        with patch("os.path.exists", return_value=False):
            _load_templates()
            assert wp._WGSL_TEMPLATES.get("templates", {}) == {}
            _load_webgpu_ops()
            assert wp._WEBGPU_OPS == {}
            assert get_wgsl_template("missing") == {}
            assert get_js_orchestration_template("missing") == ""
            assert get_wgsl_global_bindings() == ""
            assert get_webgpu_ops() == {}

        wp._WGSL_TEMPLATES = {}
        wp._WEBGPU_OPS = {}
        wp._WGSL_KERNELS = {}
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", MagicMock()),
            patch("yaml.safe_load", return_value=["not", "a", "dict"]),
        ):
            _load_templates()
            _load_webgpu_ops()
            assert wp._WGSL_TEMPLATES.get("templates", {}) == {}

        wp._WGSL_TEMPLATES = {}
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", MagicMock()),
            patch("yaml.safe_load", return_value={"templates": {}, "js_orchestration": {}, "global_bindings": None}),
        ):
            wp._merge_modular_templates(wp._WGSL_TEMPLATES)
            assert "global_bindings" not in wp._WGSL_TEMPLATES
    finally:
        wp._WGSL_TEMPLATES = {}
        wp._WEBGPU_OPS = {}
        wp._WGSL_KERNELS = {}
        _load_templates()
        _load_webgpu_ops()


def test_webgpu_generator_shader_emission_and_compile_aot() -> None:
    """Verify `WebGPUCodeGenerator.generate()` and `compile_aot()` across multi-input, conv2d, matmul, and pool ops."""
    g = IRGraph(name="test_webgpu_ops")
    g.inputs = ["x", "x2d", "x4d", "w4d"]
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 3.14}, shape_metadata=(1, 4))
    g.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(1, 4))
    g.nodes["x2d"] = IRNode(id="x2d", op_type="Input", shape_metadata=(4, 4))
    g.nodes["x4d"] = IRNode(id="x4d", op_type="Input", shape_metadata=(1, 1, 4, 4))
    g.nodes["w4d"] = IRNode(id="w4d", op_type="Input", shape_metadata=(1, 1, 3, 3))
    g.nodes["ln"] = IRNode(id="ln", op_type="LayerNorm", inputs=["x"], shape_metadata=(1, 4))
    g.nodes["mm"] = IRNode(id="mm", op_type="MatMul", inputs=["x2d", "x2d"], shape_metadata=(4, 4))
    g.nodes["conv"] = IRNode(id="conv", op_type="conv2d", inputs=["x4d", "w4d"], shape_metadata=(1, 1, 4, 4))
    g.nodes["pool"] = IRNode(
        id="pool",
        op_type="MaxPool2D",
        inputs=["x4d"],
        shape_metadata=(1, 1, 4, 4),
        attributes={"window_size": (2, 2)},
    )
    g.nodes["multi_in"] = IRNode(
        id="multi_in",
        op_type="CustomOp4",
        inputs=["c1", "x", "ln", "x"],
        shape_metadata=(1, 4),
    )
    g.outputs = ["mm", "conv", "pool", "multi_in"]

    with patch.dict(
        "ml_switcheroo_compiler.ops.registry._YAML_REGISTRY",
        {"CustomOp4": {"variants": {"edge_wgsl": {"template": "unary", "expr": "buf_in0_f32[in0_offset]"}}}},
    ):
        gen = WebGPUCodeGenerator(g)
        code = gen.generate()
        assert "compute_mm" in code
        assert "compute_conv" in code
        assert "compute_pool" in code
        assert "compute_multi_in" in code

        bundle_same = gen.compile_aot(g)
        assert isinstance(bundle_same.callable_fn, dict)
        assert bundle_same.callable_fn["status"] == "ready_to_dispatch"

        g_other = IRGraph(name="other_graph")
        g_other.inputs = ["in_other"]
        g_other.nodes["in_other"] = IRNode(id="in_other", op_type="Input", shape_metadata=(2,))
        g_other.outputs = ["in_other"]
        bundle_other = gen.compile_aot(g_other)
        assert isinstance(bundle_other.callable_fn, dict)
        assert bundle_other.callable_fn["status"] == "ready_to_dispatch"
