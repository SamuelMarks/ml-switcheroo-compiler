from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_scan_generation():
    graph = IRGraph()
    node_in = IRNode(id="node_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])

    body_graph = IRGraph()
    body_in = IRNode(id="body_in", op_type="Input", inputs=[], attributes={"dtype": "float32"}, shape_metadata=[3])
    body_add = IRNode(id="body_add", op_type="Add", inputs=["body_in", "body_in"], attributes={"dtype": "float32"}, shape_metadata=[3])
    body_graph.nodes["body_in"] = body_in
    body_graph.nodes["body_add"] = body_add
    body_graph.outputs = ["body_add"]
    body_graph.inputs = ["body_in"]

    node_scan = IRNode(id="node_scan", op_type="Scan", inputs=["node_in"], attributes={"dtype": "float32", "body_graph": body_graph}, shape_metadata=[3])

    graph.nodes["node_in"] = node_in
    graph.nodes["node_scan"] = node_scan
    graph.outputs = ["node_scan"]
    graph.inputs = ["node_in"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()

    assert "float* buf_body_add" in code
    assert "wasm_f32x4_add" in code
    assert "for (int i = 0; i < 3; ++i)" in code
