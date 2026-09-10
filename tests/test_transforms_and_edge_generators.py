"""Tests for graph transformations and edge code generators."""

from __future__ import annotations

from unittest.mock import mock_open, patch

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
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.autodiff_rules.autodiff_provider import (
    _build_parsed_call,
    _fallback_finite_difference_jvp,
    _load_autodiff_rule,
    _parse_args_and_attrs,
    _parse_expression,
    _split_nested_args,
)
from ml_switcheroo_compiler.transforms.passes.control_flow_opt import (
    control_flow_optimization_pass,
    evaluate_static_predicate,
)
from ml_switcheroo_compiler.transforms.passes.parallel_scan import (
    _load_scan_rules,
    detect_associative_reduction,
    parallel_scan_pass,
)
from ml_switcheroo_compiler.transforms.passes.vectorization import (
    _align_batch_axis,
    _load_vmap_rules,
    vectorization_pass,
    vectorize_graph,
)


def test_control_flow_opt_branches() -> None:
    """Test branch coverage in control_flow_opt.py."""
    g = IRGraph()
    # 1. Constant with string value (not bool/int/float, not having .item())
    g.nodes["c_str"] = IRNode(id="c_str", op_type="Constant", attributes={"value": "unsupported_type"})
    assert evaluate_static_predicate(g, "c_str") is None

    # 2. Non-comparison op_type
    g.nodes["add_node"] = IRNode(id="add_node", op_type="Add", inputs=["c_str"])
    assert evaluate_static_predicate(g, "add_node") is None

    # 3. Comparison op where inputs are not Constant
    g.nodes["in_var"] = IRNode(id="in_var", op_type="Input")
    g.nodes["c_num"] = IRNode(id="c_num", op_type="Constant", attributes={"value": 1})
    g.nodes["cmp_non_const"] = IRNode(id="cmp_non_const", op_type="Equal", inputs=["in_var", "c_num"])
    assert evaluate_static_predicate(g, "cmp_non_const") is None

    # 4. Comparison op where constants are non-numeric strings
    g.nodes["c_str2"] = IRNode(id="c_str2", op_type="Constant", attributes={"value": "other_str"})
    g.nodes["cmp_str"] = IRNode(id="cmp_str", op_type="Equal", inputs=["c_str", "c_str2"])
    assert evaluate_static_predicate(g, "cmp_str") is None

    # 5. Comparison op with < 2 inputs (line 34->53)
    g.nodes["cmp_single_in"] = IRNode(id="cmp_single_in", op_type="Equal", inputs=["c_num"])
    assert evaluate_static_predicate(g, "cmp_single_in") is None

    # 6. Comparison op where one is int and other is string (line 50->53)
    g.nodes["cmp_mixed"] = IRNode(id="cmp_mixed", op_type="Equal", inputs=["c_num", "c_str"])
    assert evaluate_static_predicate(g, "cmp_mixed") is None

    # GreaterEqual comparison (line 50-51)
    g.nodes["c_5"] = IRNode(id="c_5", op_type="Constant", attributes={"value": 5})
    g.nodes["c_3"] = IRNode(id="c_3", op_type="Constant", attributes={"value": 3})
    g.nodes["cmp_ge"] = IRNode(id="cmp_ge", op_type="GreaterEqual", inputs=["c_5", "c_3"])
    assert evaluate_static_predicate(g, "cmp_ge") is True

    # 7. control_flow_optimization_pass with active_branch = None and hoisted = []
    g_opt = IRGraph()
    g_opt.nodes["pred_const"] = IRNode(id="pred_const", op_type="Constant", attributes={"value": True})
    f_block = IRGraph()
    f_block.nodes["fn"] = IRNode(id="fn", op_type="Identity", inputs=[])
    g_opt.nodes["cond_node"] = IRNode(
        id="cond_node",
        op_type="Cond",
        inputs=["pred_const"],
        attributes={"true_branch": None, "false_branch": f_block},
    )
    res_graph = control_flow_optimization_pass(g_opt)
    assert res_graph is not None

    # 8. Step 2 invariant hoisting where hoisted is empty
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
        attributes={"true_branch": b1, "false_branch": b2},
    )
    res_no_hoist = control_flow_optimization_pass(g_no_hoist)
    assert res_no_hoist is not None


def test_parallel_scan_branches() -> None:
    """Test branch coverage in parallel_scan.py."""
    # 1. _load_scan_rules with non-dict YAML data
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="['not_a_dict']")):
            with patch("yaml.safe_load", return_value=["not_a_dict"]):
                assert _load_scan_rules() == {}

    # 2. detect_associative_reduction: op not in assoc_map
    b_graph = IRGraph()
    b_graph.nodes["unknown_op"] = IRNode(id="unknown_op", op_type="CustomUnsupportedOp")
    rules = {"associative_ops": {"Add": {"parallel_primitive": "CumSum"}}}
    assert detect_associative_reduction(b_graph, rules) is None

    # 3. parallel_scan_pass: node is Scan but reduction is not associative
    g_scan = IRGraph()
    g_scan.nodes["scan_unassoc"] = IRNode(
        id="scan_unassoc",
        op_type="Scan",
        inputs=["init", "xs"],
        attributes={"body": b_graph},
    )
    res_scan = parallel_scan_pass(g_scan)
    assert "scan_unassoc" in res_scan.nodes

    # 4. parallel_scan_pass: transformed=False (no Scan nodes in graph)
    g_empty = IRGraph()
    g_empty.nodes["noop"] = IRNode(id="noop", op_type="Identity")
    assert parallel_scan_pass(g_empty) == g_empty


def test_vectorization_branches() -> None:
    """Test branch coverage in vectorization.py."""
    # 1. _load_vmap_rules when data is not dict or rules missing
    with patch("os.path.exists", return_value=True):
        with patch("yaml.safe_load", return_value={"no_rules_key": 1}):
            assert _load_vmap_rules() == {}

    # 2. _align_batch_axis when inp_id not in graph.nodes
    g = IRGraph()
    res_align = _align_batch_axis(g, "missing_node", current_axis=1, target_axis=0, shape=(2, 4))
    assert res_align is not None

    # 3. vectorize_graph when input node not in graph.nodes (line 141->143) and unmapped Input node (line 155->157)
    g_missing_inp = IRGraph()
    g_missing_inp.inputs = ["ghost_input"]
    g_missing_inp.nodes["unmapped_input"] = IRNode(id="unmapped_input", op_type="Input")
    g_missing_inp.outputs = ["unmapped_input"]
    v_missing = vectorize_graph(g_missing_inp, in_axes=0, batch_size=2)
    assert "unmapped_input" in v_missing.nodes

    # 4. vectorize_graph policies: shift_axis with negative axis and dim
    g_shift = IRGraph()
    g_shift.inputs = ["x"]
    g_shift.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
    g_shift.nodes["red"] = IRNode(id="red", op_type="ReduceSum", inputs=["x"], attributes={"axis": -1, "dim": -1})
    g_shift.outputs = ["red"]
    v_shift = vectorize_graph(g_shift, in_axes=0, batch_size=2)
    assert "red" in v_shift.nodes

    # 5. prepend_batch_dim with shape and newshape (line 217->219)
    g_reshape = IRGraph()
    g_reshape.inputs = ["x"]
    g_reshape.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(4,))
    g_reshape.nodes["resh1"] = IRNode(id="resh1", op_type="Reshape", inputs=["x"], attributes={"shape": (2, 2)})
    g_reshape.nodes["resh2"] = IRNode(id="resh2", op_type="Reshape", inputs=["x"], attributes={"newshape": (2, 2)})
    g_reshape.outputs = ["resh1", "resh2"]
    v_resh = vectorize_graph(g_reshape, in_axes=0, batch_size=2)
    assert "resh1" in v_resh.nodes and "resh2" in v_resh.nodes

    # 6. Determine batch_size in vectorization_pass where axis >= len(shape) (line 281->284) and ghost input (line 278->284)
    g_vmap_wrapper = IRGraph()
    g_vmap_wrapper.nodes["inp_short_shape"] = IRNode(id="inp_short_shape", op_type="Input", shape_metadata=(4,))
    g_vmap_wrapper.nodes["vmap_node"] = IRNode(
        id="vmap_node",
        op_type="Vmap",
        inputs=["inp_short_shape"],
        attributes={"in_axes": 5, "body": g_shift},
    )
    g_vmap_wrapper.nodes["vmap_ghost"] = IRNode(
        id="vmap_ghost",
        op_type="Vmap",
        inputs=["ghost_not_in_graph"],
        attributes={"in_axes": 0, "body": g_shift},
    )
    v_wrapper = vectorization_pass(g_vmap_wrapper)
    assert "vmap_node" in v_wrapper.nodes and "vmap_ghost" in v_wrapper.nodes


def test_autodiff_provider_branches() -> None:
    """Test branch coverage in autodiff_provider.py."""
    # 1. _split_nested_args with empty string
    assert _split_nested_args("") == []

    # 2. _parse_args_and_attrs with invalid python literal attribute
    g = IRGraph()
    node = IRNode(id="n1", op_type="CustomOp")
    inputs, attrs = _parse_args_and_attrs(g, ["foo=invalid_unquoted_literal"], node)
    assert attrs.get("foo") == "invalid_unquoted_literal"

    # 3. BroadcastReduce (lines 105-107) and BroadcastLike (lines 110-112)
    node_br = IRNode(id="orig_node", op_type="Mul", inputs=["a", "b"])
    g.nodes["a"] = IRNode(id="a", op_type="Input", shape_metadata=(2, 4))
    g.nodes["b"] = IRNode(id="b", op_type="Input", shape_metadata=(2, 4))
    br_id = _build_parsed_call(g, "BroadcastReduce", ["$input[0]", "$input[1]"], node_br)
    assert br_id in g.nodes
    bl_id = _build_parsed_call(g, "BroadcastLike", ["$input[0]", "$input[1]"], node_br)
    assert bl_id in g.nodes

    # 4. SetItem when node.op_type != "SetItem" (line 116)
    node_non_set = IRNode(id="node_non_set", op_type="DifferentOp", inputs=["a"], attributes={"indexed": True})
    set_id = _build_parsed_call(g, "SetItem", ["$input[0]"], node_non_set)
    assert set_id in g.nodes

    # 5. _parse_expression starting with negative sign "- "
    neg_id = _parse_expression(g, "- $output", node)
    assert neg_id is not None

    # 6. _fallback_finite_difference_jvp where len(tangents) < len(node.inputs)
    node_multi_in = IRNode(id="n_multi", op_type="Add", inputs=["in1", "in2"])
    g.nodes["in1"] = IRNode(id="in1", op_type="Input")
    g.nodes["in2"] = IRNode(id="in2", op_type="Input")
    jvp_id = _fallback_finite_difference_jvp(g, node_multi_in, ["tan1"])
    assert jvp_id in g.nodes

    # 7. _load_autodiff_rule fallback to autodiff_rules.yaml (lines 241->244 and 242)
    with patch("os.path.exists") as mock_exists:
        mock_exists.side_effect = lambda p: p.endswith("autodiff_rules.yaml")
        with patch("builtins.open", mock_open(read_data="LegacyFoundOp:\n  vjp: {rule: 1}\n")):
            with patch("yaml.safe_load", return_value={"LegacyFoundOp": {"vjp": {"rule": 1}}}):
                assert _load_autodiff_rule("MissingOp", "vjp") is None
                assert _load_autodiff_rule("LegacyFoundOp", "vjp") == {"vjp": {"rule": 1}}

    # 8. ExpandDims/Squeeze and Reshape in _build_parsed_call (lines 129, 134)
    sq_node = IRNode(id="sq_n", op_type="Squeeze", inputs=["a"], attributes={"axis": 1})
    g.nodes["sq_n"] = sq_node
    exp_call_id = _build_parsed_call(g, "ExpandDims", ["$input[0]"], sq_node)
    assert exp_call_id in g.nodes

    exp_node = IRNode(id="exp_n", op_type="ExpandDims", inputs=["a"], attributes={"axis": 1})
    g.nodes["exp_n"] = exp_node
    sq_call_id = _build_parsed_call(g, "Squeeze", ["$input[0]"], exp_node)
    assert sq_call_id in g.nodes

    reshape_node = IRNode(id="r_node", op_type="Reshape", inputs=["a"])
    g.nodes["r_node"] = reshape_node
    res_call_id = _build_parsed_call(g, "Reshape", ["$input[0]"], reshape_node)
    assert res_call_id in g.nodes
    assert g.nodes[res_call_id].shape_metadata == (2, 4)


def test_webgl_custom_setup_empty_inputs() -> None:
    """Test WebGLCodeGenerator._emit_node with custom_setup and empty node.inputs."""
    g = IRGraph(name="webgl_empty_inputs")
    node = IRNode(id="test_node", op_type="matmul", inputs=[], shape_metadata=(16, 16))
    g.nodes["test_node"] = node
    g.outputs = ["test_node"]

    gen = WebGLCodeGenerator(g)
    js = []
    w, h = gen._emit_node(node, {}, js)
    assert w > 0 and h > 0
    assert any("kDim" in line or "32" in line for line in js)


def test_wasm_provider_branches(tmp_path) -> None:
    """Test load_yaml_dir branches with empty dictionary and non-dict values."""
    yaml_dir = tmp_path / "yaml_test_dir"
    yaml_dir.mkdir()

    f1 = str(yaml_dir / "empty.yaml")
    with open(f1, "w") as f:
        f.write("templates: {}\n")

    f2 = str(yaml_dir / "list_val.yaml")
    with open(f2, "w") as f:
        f.write("templates:\n  raw_list:\n    - item1\n    - item2\n")

    with patch("pathlib.Path.is_dir", return_value=True):
        with patch("glob.glob", return_value=[f1, f2]):
            res = load_yaml_dir("yaml_test_dir")
            assert "templates" in res
            assert "raw_list" in res["templates"]


def test_wgsl_provider_missing_files() -> None:
    """Test wgsl_provider when templates or ops files do not exist."""
    from unittest.mock import MagicMock

    import ml_switcheroo_compiler.backends.edge.wgsl.wgsl_provider as wp

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

    # Non-dict YAML data
    wp._WGSL_TEMPLATES = {}
    wp._WEBGPU_OPS = {}
    wp._WGSL_KERNELS = {}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value=["not", "a", "dict"]):
        _load_templates()
        _load_webgpu_ops()
        assert wp._WGSL_TEMPLATES.get("templates", {}) == {}

    # Modular template without global_bindings (line 31->exit)
    wp._WGSL_TEMPLATES = {}
    with patch("os.path.exists", return_value=True), patch("builtins.open", MagicMock()), patch("yaml.safe_load", return_value={"templates": {}, "js_orchestration": {}, "global_bindings": None}):
        wp._merge_modular_templates(wp._WGSL_TEMPLATES)
        assert "global_bindings" not in wp._WGSL_TEMPLATES

    wp._WGSL_TEMPLATES = {}
    wp._WEBGPU_OPS = {}
    wp._WGSL_KERNELS = {}
    _load_templates()
    _load_webgpu_ops()


def test_webgpu_generator_branches() -> None:
    """Test WebGPUCodeGenerator branches: op templates, 4 inputs for j>=3, and _compile_aot_impl."""
    # 1. Test op types in generate
    g = IRGraph(name="test_webgpu_ops")
    g.nodes["c1"] = IRNode(id="c1", op_type="Constant", attributes={"value": 3.14})
    g.nodes["x"] = IRNode(id="x", op_type="Input", shape_metadata=(1, 4))
    g.nodes["ln"] = IRNode(id="ln", op_type="LayerNorm", inputs=["x"], shape_metadata=(1, 4))

    # Op with 4 inputs to trigger j >= 3 branch (line 545->544)
    g.nodes["multi_in"] = IRNode(
        id="multi_in",
        op_type="CustomOp4",
        inputs=["c1", "x", "ln", "x"],
        shape_metadata=(1, 4),
    )
    g.outputs = ["multi_in"]

    with patch.dict("ml_switcheroo_compiler.ops.registry._YAML_REGISTRY", {"CustomOp4": {"variants": {"edge_wgsl": {"template": "unary", "expr": "buf_in0_f32[in0_offset]"}}}}):
        gen = WebGPUCodeGenerator(g)
        code = gen.generate()
        assert "compute_" in code

        # 2. Call _get_wgsl_for_op directly for MatMul and Conv2D to execute lines 160 and 162
        g.nodes["x4d"] = IRNode(id="x4d", op_type="Input", shape_metadata=(1, 1, 4, 4))
        g.nodes["w4d"] = IRNode(id="w4d", op_type="Input", shape_metadata=(1, 1, 3, 3))
        gen.sorted_nodes.extend([g.nodes["x4d"], g.nodes["w4d"]])

        mm_node = IRNode(id="mm", op_type="MatMul", inputs=["x", "x"], shape_metadata=(4, 4))
        wgsl_mm, _, _, _ = gen._get_wgsl_for_op(mm_node, (4, 4), 16, "mm")
        assert wgsl_mm is not None

        conv_node = IRNode(id="conv", op_type="Conv2D", inputs=["x4d", "w4d"], shape_metadata=(1, 1, 4, 4))
        wgsl_conv, _, _, _ = gen._get_wgsl_for_op(conv_node, (1, 1, 4, 4), 16, "conv")
        assert wgsl_conv is not None

        pool_node = IRNode(id="pool", op_type="MaxPool2D", inputs=["x4d"], shape_metadata=(1, 1, 4, 4), attributes={"window_size": (2, 2)})
        wgsl_pool, _, _, _ = gen._get_wgsl_for_op(pool_node, (1, 1, 4, 4), 16, "pool")
        assert wgsl_pool is not None

        # 3. _compile_aot_impl with graph == self.graph and graph != self.graph
        bundle_same = gen._compile_aot_impl(g)
        assert bundle_same["status"] == "ready_to_dispatch"

        g_other = IRGraph(name="other_graph")
        g_other.nodes["in_other"] = IRNode(id="in_other", op_type="Input", shape_metadata=(2,))
        g_other.outputs = ["in_other"]
        bundle_other = gen._compile_aot_impl(g_other)
        assert bundle_other["status"] == "ready_to_dispatch"
