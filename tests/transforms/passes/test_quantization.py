from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.transforms.passes.quantization import QATFakeQuantizePass, QuantizationConfig

"""Tests for quantization pass."""

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.quantization import PTQPass


def test_ptq_pass() -> None:
    """Test that the PTQ pass annotates Dot operations."""
    config = QuantizationConfig(target_dtype=DType.Int8, per_channel=True, symmetric=True)
    dataset = Dataset()

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "weight": IRNode(id="weight", op_type="Constant", inputs=[]),
        "dot": IRNode(id="dot", op_type="Dot", inputs=["input", "weight"]),
        "add": IRNode(id="add", op_type="Add", inputs=["dot", "dot"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])

    ptq = PTQPass(config, dataset)
    optimized_graph = ptq(graph)

    assert "ptq_target_dtype" in optimized_graph.nodes["dot"].attributes
    assert optimized_graph.nodes["dot"].attributes["ptq_target_dtype"] == "Int8"
    assert optimized_graph.nodes["dot"].attributes["ptq_per_channel"] is True
    assert optimized_graph.nodes["dot"].attributes["ptq_symmetric"] is True

    assert "ptq_target_dtype" not in optimized_graph.nodes["add"].attributes


def test_ptq_pass_no_op() -> None:
    """Test PTQ pass when no ops apply."""
    config = QuantizationConfig(target_dtype=DType.Int8)
    dataset = Dataset()

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])

    ptq = PTQPass(config, dataset)
    optimized_graph = ptq(graph)
    assert "ptq_target_dtype" not in optimized_graph.nodes["add"].attributes


from ml_switcheroo_compiler.transforms.passes.quantization import IntegerQuantizationLoweringPass, PTQCalibrationPass


def test_ptq_calibration_pass() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=False)
    dataset = Dataset()

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul"])

    # Test minmax
    pass_obj = PTQCalibrationPass(config, dataset, method="minmax")
    assert pass_obj(graph) is True

    assert "calibration_min" in graph.nodes["matmul"].attributes
    assert graph.nodes["matmul"].attributes["calibration_min"] == 0.0
    assert graph.nodes["matmul"].attributes["calibration_max"] == 1.0

    # Test histogram
    pass_obj_hist = PTQCalibrationPass(config, dataset, method="histogram")
    assert pass_obj_hist(graph) is True
    assert "calibration_histogram" in graph.nodes["matmul"].attributes


def test_ptq_calibration_no_op() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8)
    dataset = Dataset()
    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])
    pass_obj = PTQCalibrationPass(config, dataset)
    assert pass_obj(graph) is False


def test_qat_fake_quantize_pass() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=True)

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "conv": IRNode(id="conv", op_type="Conv2D", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["conv"])

    pass_obj = QATFakeQuantizePass(config)
    assert pass_obj(graph) is True

    assert "input_fake_quant" in graph.nodes
    assert graph.nodes["input_fake_quant"].op_type == "FakeQuantize"
    assert graph.nodes["input_fake_quant"].attributes["bits"] == 8
    assert graph.nodes["conv"].inputs == ["input_fake_quant", "input_fake_quant"]


def test_qat_fake_quantize_no_op() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=True)
    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "add": IRNode(id="add", op_type="Add", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])
    pass_obj = QATFakeQuantizePass(config)
    assert pass_obj(graph) is False


def test_integer_quantization_lowering_pass() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=False)

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["input", "input"], attributes={"calibration_min": 0.0}),
        "conv": IRNode(id="conv", op_type="Conv2D", inputs=["input", "input"], attributes={"calibration_min": 0.0}),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul", "conv"])

    pass_obj = IntegerQuantizationLoweringPass(config)
    assert pass_obj(graph) is True

    assert graph.nodes["matmul"].op_type == "QuantizedMatMul"
    assert graph.nodes["matmul"].attributes["q_zero_point"] == 128

    assert graph.nodes["conv"].op_type == "QuantizedConv2D"
    assert graph.nodes["conv"].attributes["dtype"] == "Int8"


def test_integer_quantization_lowering_no_op() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=True)
    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["input", "input"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul"])
    pass_obj = IntegerQuantizationLoweringPass(config)
    assert pass_obj(graph) is False


def test_qat_pass_no_new_inputs():
    g = IRGraph()
    n3 = IRNode(id="n3", op_type="MatMul", inputs=[])
    g.nodes["n3"] = n3
    assert QATFakeQuantizePass(QuantizationConfig(target_dtype=DType.Int8, per_channel=False, symmetric=True))(g) is False
