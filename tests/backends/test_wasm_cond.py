from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_cond_generation():
    graph = IRGraph()
    node_in = IRNode(id="node_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])

    # Create true branch graph
    true_graph = IRGraph()
    true_in = IRNode(id="true_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])
    true_add = IRNode(id="true_add", op_type="Add", inputs=["true_in", "true_in"], attributes={"dtype": "float32"}, shape_metadata=[3])
    true_graph.nodes["true_in"] = true_in
    true_graph.nodes["true_add"] = true_add
    true_graph.outputs = ["true_add"]
    true_graph.inputs = ["true_in"]

    # Create false branch graph
    false_graph = IRGraph()
    false_in = IRNode(id="false_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])
    false_sub = IRNode(id="false_sub", op_type="Subtract", inputs=["false_in", "false_in"], attributes={"dtype": "float32"}, shape_metadata=[3])
    false_graph.nodes["false_in"] = false_in
    false_graph.nodes["false_sub"] = false_sub
    false_graph.outputs = ["false_sub"]
    false_graph.inputs = ["false_in"]

    node_cond = IRNode(id="node_cond", op_type="Cond", inputs=["node_in", "node_in", "node_in"], attributes={"dtype": "float32", "branch_graphs": [true_graph, false_graph]}, shape_metadata=[3])

    graph.nodes["node_in"] = node_in
    graph.nodes["node_cond"] = node_cond
    graph.outputs = ["node_cond"]
    graph.inputs = ["node_in"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "buf_true_add" in code
    assert "wasm_f32x4_add" in code
    assert "buf_false_sub" in code
    assert "wasm_f32x4_sub" in code
