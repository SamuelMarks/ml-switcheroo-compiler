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
