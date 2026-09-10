"""Tests for WASM branch generation and edge cases."""


def test_wasm_cond_missing_inputs():
    """Test WASM branch generation when inputs are missing."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    branch_graph = IRGraph()
    branch_graph.inputs = ["missing_input"]  # Missing from parent inputs

    node = IRNode("cond_1", "Cond", inputs=[], shape_metadata=[2, 2])
    node.attributes = {"then_branch": branch_graph}

    subnode = IRNode("add_1", "Add", inputs=["missing_input", "other_input"], shape_metadata=[2, 2])
    branch_graph.nodes["add_1"] = subnode
    branch_graph.sorted_nodes = [subnode]

    generator = WasmCodeGenerator(graph, [])

    generator._generate_op(node, "Cond", "cond_1", [], [2, 2], 4)


def test_wasm_in0_shape():
    """Test WASM shape parsing with edge cases."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    in0 = IRNode("in0", "Placeholder", inputs=[], shape_metadata=[])
    in1 = IRNode("in1", "Placeholder", inputs=[], shape_metadata=1)

    node = IRNode("cust", "Custom", inputs=["in0"], shape_metadata=[1, 1])
    node2 = IRNode("cust2", "Custom", inputs=["in1"], shape_metadata=[1, 1])

    graph.nodes["in0"] = in0
    graph.nodes["in1"] = in1
    graph.nodes["cust"] = node
    graph.nodes["cust2"] = node2
    graph.sorted_nodes = [in0, in1, node, node2]
    generator = WasmCodeGenerator(graph, [])

    with patch.dict("ml_switcheroo_compiler.ops.generated_registry.OPS_REGISTRY", {"Custom": {"variants": {"edge_wasm_simd": {"template": "custom", "body": "a+b"}}}}):
        with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value={"body": "return a+b;"}):
            generator._generate_op(node, "Custom", "cust", ["in0"], [1, 1], 1)
            generator._generate_op(node2, "Custom", "cust2", ["in1"], [1, 1], 1)


def test_wasm_simd_helpers_branches():
    """Test branches in get_helper_functions."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph

    gen = WasmCodeGenerator(IRGraph())

    # 107->127: yaml_path does not exist
    with patch("os.path.exists", return_value=False):
        helpers = gen.get_helper_functions()
        assert isinstance(helpers, list)

    # 120->119: intrinsic missing macro_name or simd_expr
    mock_data = {
        "scalars": {},
        "intrinsics": {
            "incomplete_op": {"macro_name": "incomplete"}  # missing simd_expr
        },
    }
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            mock_open.return_value.__enter__.return_value = "file"
            with patch("yaml.safe_load", return_value=mock_data):
                helpers = gen.get_helper_functions()
                assert isinstance(helpers, list)


def test_wasm_while_loop_and_scan_empty_body():
    """Test while_loop and scan when body/body_graph is missing."""
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    gen = WasmCodeGenerator(IRGraph())

    # 267->271: WhileLoop with no body_graph
    while_node = IRNode("w1", "WhileLoop", inputs=["in_0"], attributes={})
    gen.visit_WhileLoop(while_node, "WhileLoop", "w1", ["in_0"], [1], 1)

    # visit_Broadcast
    bcast_node = IRNode("bc1", "Broadcast", inputs=["in_0"], shape_metadata=[1])
    gen.visit_Broadcast(bcast_node, "Broadcast", "bc1", ["in_0"], [1], 1)
    gen.visit_Broadcast(bcast_node, "Broadcast", "bc2", [], [1], 1)

    # 349->353: Scan with no body_graph
    scan_node = IRNode("s1", "Scan", inputs=["in_0"], attributes={})
    gen.visit_Scan(scan_node, "Scan", "s1", ["in_0"], [1], 1)


def test_wasm_templates_whitespace_lines():
    """Test whitespace-only lines in Linear, Attention, LayerNorm, ReduceSum templates."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    graph = IRGraph()
    gen = WasmCodeGenerator(graph)

    whitespace_template = {"body": "int x = 1;\n   \nint y = 2;"}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.get_wasm_template", return_value=whitespace_template):
        # 478->477: Linear
        linear_node = IRNode("lin", "Linear", inputs=["x", "w"], shape_metadata=[2, 2])
        gen.visit_Linear(linear_node, "Linear", "lin", ["x", "w"], [2, 2], 4)

        # 497->496: Attention
        attn_node = IRNode("attn", "Attention", inputs=["q", "k", "v"], shape_metadata=[2, 4])
        gen.visit_Attention(attn_node, "Attention", "attn", ["q", "k", "v"], [2, 4], 8)

        # 523->522: LayerNorm
        ln_node = IRNode("ln", "LayerNorm", inputs=["x", "g", "b"], shape_metadata=[2, 4])
        gen.visit_LayerNorm(ln_node, "LayerNorm", "ln", ["x", "g", "b"], [2, 4], 8)

        # 555->554: ReduceSum
        red_node = IRNode("red", "ReduceSum", inputs=["x"], shape_metadata=[2, 2])
        gen.visit_ReduceSum(red_node, "ReduceSum", "red", ["x"], [2, 2], 4)


def test_wasm_vector_unrolled_branches():
    """Test _generate_vector_unrolled_op edge cases."""
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    gen = WasmCodeGenerator(IRGraph())
    node = IRNode("op1", "Sin", inputs=["in_0"], shape_metadata=[4])

    # 572->581: yaml_path does not exist
    with patch("os.path.exists", return_value=False):
        gen._generate_vector_unrolled_op(node, "Sin", "op1", ["in_0"], [4], 4)

    # 578->581: intr exists but has no scalar_fallback
    mock_data = {
        "scalars": {},
        "intrinsics": {
            "Sin": {"macro_name": "wasm_f32x4_sin"}  # no scalar_fallback
        },
    }
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", new_callable=MagicMock) as mock_open:
            mock_open.return_value.__enter__.return_value = "file"
            with patch("yaml.safe_load", return_value=mock_data):
                gen._generate_vector_unrolled_op(node, "Sin", "op1", ["in_0"], [4], 4)
