from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_wgsl_generator():
    graph = IRGraph()
    graph.nodes = {"x": IRNode("x", "Input", inputs=[], shape_metadata=[10]), "y": IRNode("y", "Input", inputs=[], shape_metadata=[10]), "add": IRNode("add", "Add", inputs=["x", "y"], shape_metadata=[10])}
    graph.inputs = ["x", "y"]
    graph.outputs = ["add"]

    gen = WebGPUCodeGenerator(graph)
    code = gen.generate()
    assert "buf_out_f32" in code
    assert "createBuffer" in code
    assert "dispatchWorkgroups" in code
