from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.operator_fusion import apply_operator_fusion
from ml_switcheroo_compiler.transforms.passes.quantization import IntegerQuantizationLoweringPass, PTQPass, QATFakeQuantizePass, QuantizationConfig


def test_operator_fusion_mockups():
    # Linear fusion (MatMul + Add)
    graph = IRGraph()
    graph.nodes = {"x": IRNode("x", "Input", inputs=[]), "w": IRNode("w", "Input", inputs=[]), "matmul": IRNode("matmul", "MatMul", inputs=["x", "w"]), "b": IRNode("b", "Input", inputs=[]), "add": IRNode("add", "Add", inputs=["matmul", "b"])}
    graph.inputs = ["x", "w", "b"]
    graph.outputs = ["add"]

    assert len(graph.nodes) == 5
    g2 = apply_operator_fusion(graph)
    assert len(g2.nodes) == 4
    assert "add" in g2.nodes
    assert g2.nodes["add"].op_type == "Linear"

    # Conv2D + BatchNorm fusion
    graph_conv = IRGraph()
    graph_conv.nodes = {
        "x": IRNode("x", "Input", inputs=[]),
        "w": IRNode("w", "Input", inputs=[]),
        "conv": IRNode("conv", "Conv2D", inputs=["x", "w"]),
        "gamma": IRNode("gamma", "Input", inputs=[]),
        "beta": IRNode("beta", "Input", inputs=[]),
        "mean": IRNode("mean", "Input", inputs=[]),
        "var": IRNode("var", "Input", inputs=[]),
        "bn": IRNode("bn", "BatchNorm", inputs=["conv", "gamma", "beta", "mean", "var"]),
    }
    graph_conv.inputs = ["x", "w", "gamma", "beta", "mean", "var"]
    graph_conv.outputs = ["bn"]

    g3 = apply_operator_fusion(graph_conv)
    assert len(g3.nodes) == 7
    assert g3.nodes["bn"].op_type == "Conv2DBatchNorm"

    # Dot + Relu fusion
    graph_dot = IRGraph()
    graph_dot.nodes = {"x": IRNode("x", "Input", inputs=[]), "y": IRNode("y", "Input", inputs=[]), "dot": IRNode("dot", "Dot", inputs=["x", "y"]), "relu": IRNode("relu", "Relu", inputs=["dot"])}
    graph_dot.inputs = ["x", "y"]
    graph_dot.outputs = ["relu"]

    g4 = apply_operator_fusion(graph_dot)
    assert len(g4.nodes) == 3
    assert g4.nodes["relu"].op_type == "DotRelu"


def test_quantization_mockups():
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=False)

    # PTQPass
    graph = IRGraph()
    graph.nodes = {"x": IRNode("x", "Input", inputs=[]), "w": IRNode("w", "Input", inputs=[]), "matmul": IRNode("matmul", "MatMul", inputs=["x", "w"])}
    graph.inputs = ["x", "w"]
    graph.outputs = ["matmul"]

    PTQPass(config, None)(graph)
    assert graph.nodes["matmul"].op_type == "QuantizedMatMul"
    assert graph.nodes["matmul"].attributes["q_scale"] == 0.1
    assert graph.nodes["matmul"].attributes["q_zero_point"] == 128

    # QATFakeQuantizePass
    graph2 = IRGraph()
    graph2.nodes = {"x": IRNode("x", "Input", inputs=[]), "w": IRNode("w", "Input", inputs=[]), "matmul": IRNode("matmul", "MatMul", inputs=["x", "w"])}
    graph2.inputs = ["x", "w"]
    graph2.outputs = ["matmul"]

    QATFakeQuantizePass(config)(graph2)
    assert "x_fake_quant" in graph2.nodes
    assert "w_fake_quant" in graph2.nodes
    assert graph2.nodes["x_fake_quant"].op_type == "FakeQuantize"
    assert graph2.nodes["w_fake_quant"].attributes["bits"] == 8

    # IntegerQuantizationLoweringPass
    graph3 = IRGraph()
    graph3.nodes = {"x": IRNode("x", "Input", inputs=[]), "w": IRNode("w", "Input", inputs=[]), "matmul": IRNode("matmul", "MatMul", inputs=["x", "w"], attributes={"calibration_min": -1.0})}
    graph3.inputs = ["x", "w"]
    graph3.outputs = ["matmul"]

    IntegerQuantizationLoweringPass(config)(graph3)
    assert graph3.nodes["matmul"].op_type == "QuantizedMatMul"
    assert graph3.nodes["matmul"].attributes["dtype"] == "Int8"
