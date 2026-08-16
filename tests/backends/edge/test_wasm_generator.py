from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wasm_generator():
    pass

    pass

    graph = IRGraph()
    graph.nodes = {"x": IRNode("x", "Input", inputs=[], shape_metadata=[10]), "y": IRNode("y", "Input", inputs=[], shape_metadata=[10]), "add": IRNode("add", "Add", inputs=["x", "y"], shape_metadata=[10])}
    graph.inputs = ["x", "y"]
    graph.outputs = ["add"]

    gen = WasmCodeGenerator(graph)
    code = gen.generate()
    assert "std::aligned_alloc" in code or "out_" in code
    assert "buf_add[j]" in code
    assert "wasm_f32x4_add" in code or "scalar_expr" in code
