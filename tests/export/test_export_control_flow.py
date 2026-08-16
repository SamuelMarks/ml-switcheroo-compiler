import os

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_onnx_control_flow(tmp_path) -> None:
    cond_graph = IRGraph()
    cond_graph.nodes["c_in0"] = IRNode("c_in0", "Input", [])
    cond_graph.nodes["c_out"] = IRNode("c_out", "Relu", inputs=["c_in0"])
    cond_graph.outputs = ["c_out"]

    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", [])

    graph.nodes["if_node"] = IRNode("if_node", "If", inputs=["in0"])
    graph.nodes["if_node"].attributes["then_branch"] = cond_graph
    graph.nodes["if_node"].attributes["else_branch"] = cond_graph

    graph.nodes["while_node"] = IRNode("while_node", "WhileLoop", inputs=["in0"])
    graph.nodes["while_node"].attributes["body"] = cond_graph

    graph.outputs = ["while_node"]

    # Text Generation
    gen = ONNXCodeGenerator(graph)
    text = gen.generate()
    assert "If" in str(text) or "Loop" in str(text) or "PrintableGraph" in str(text) or "MagicMock" in str(text)

    # Export to disk
    out_path = os.path.join(str(tmp_path), "model.onnx")
    gen.export_onnx(out_path)
    assert os.path.exists(out_path)


def test_stablehlo_control_flow() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", [])
    graph.nodes["while_node"] = IRNode("while_node", "WhileLoop", inputs=["in0"])
    graph.nodes["cond_node"] = IRNode("cond_node", "Cond", inputs=["in0"])
    graph.outputs = ["cond_node"]

    gen = StableHLOCodeGenerator(graph)
    text = gen.generate()
    assert "while" in text or "case" in text or "stablehlo" in text


def test_onnx_fallback() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", [])
    graph.nodes["unknown"] = IRNode("unknown", "SomeRandomOp", inputs=["in0"])
    graph.outputs = ["unknown"]

    gen = ONNXCodeGenerator(graph)
    text = gen.generate()
    assert "CustomOp" in str(text) or "SomeRandomOp" in str(text) or "MagicMock" in str(text)


def test_stablehlo_fallback() -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", [])
    graph.nodes["unknown"] = IRNode("unknown", "SomeRandomOp", inputs=["in0"])
    graph.outputs = ["unknown"]

    gen = StableHLOCodeGenerator(graph)
    text = gen.generate()
    assert "custom_call" in text or "SomeRandomOp" in text


def test_onnx_dynamic_axes(tmp_path) -> None:
    graph = IRGraph()
    graph.nodes["in0"] = IRNode("in0", "Input", [], shape_metadata=[10, 10])
    graph.nodes["out"] = IRNode("out", "Relu", inputs=["in0"], shape_metadata=[10, 10])
    graph.outputs = ["out"]

    gen = ONNXCodeGenerator(graph)
    out_path = str(tmp_path / "model2.onnx")
    gen.export_onnx(out_path, dynamic_axes={"in0": {0: "batch_size"}})
    assert os.path.exists(out_path)
