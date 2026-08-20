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
        "conv": IRNode(id="conv", op_type="Conv2D", inputs=["input", "weight"]),
        "add": IRNode(id="add", op_type="Add", inputs=["dot", "conv"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["add"])

    ptq = PTQPass(config, dataset)
    optimized_graph = ptq(graph)

    assert "dtype" in optimized_graph.nodes["dot"].attributes
    assert optimized_graph.nodes["dot"].attributes["dtype"] == "Int8"
    assert "q_scale" in optimized_graph.nodes["dot"].attributes

    assert "dtype" not in optimized_graph.nodes["add"].attributes


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


def test_integer_quantization_drops_fake_quant() -> None:
    config = QuantizationConfig(target_dtype=DType.Int8, symmetric=False)

    nodes = {
        "input": IRNode(id="input", op_type="Input", inputs=[]),
        "fq": IRNode(id="fq", op_type="FakeQuantize", inputs=["input"]),
        "matmul": IRNode(id="matmul", op_type="MatMul", inputs=["fq", "fq"]),
    }
    graph = IRGraph(name="test", nodes=nodes, outputs=["matmul"])

    pass_obj = IntegerQuantizationLoweringPass(config)
    assert pass_obj(graph) is True

    # FakeQuantize should be dropped
    assert "fq" not in graph.nodes

    # matmul should be updated
    assert graph.nodes["matmul"].op_type == "QuantizedMatMul"
    # its inputs should bypass 'fq' to 'input'
    assert graph.nodes["matmul"].inputs == ["input", "input"]


def test_quantization_missing_coverage():
    # Hit os.path.exists == False for all passes
    from unittest.mock import MagicMock, mock_open, patch

    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
    from ml_switcheroo_compiler.transforms.passes.quantization import IntegerQuantizationLoweringPass, PTQPass, QATFakeQuantizePass

    with patch("os.path.exists", return_value=False):
        p1 = PTQPass(MagicMock(), MagicMock())
        p2 = QATFakeQuantizePass(MagicMock())
        p3 = IntegerQuantizationLoweringPass(MagicMock())
        assert p1.rules == {}
        assert p2.rules == {}
        assert p3.rules == {}

    # Hit 266 in QuantizationLoweringPass: new_inputs != node.inputs
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Dequantize", inputs=["some_input"])
    n2 = IRNode(id="n2", op_type="UnknownOp", inputs=["n1"])
    graph.nodes = {"some_input": IRNode(id="some_input", op_type="Input", inputs=[]), "n1": n1, "n2": n2}
    graph.inputs = ["some_input"]
    graph.outputs = ["n2"]

    with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open()):
        p3_b = IntegerQuantizationLoweringPass(MagicMock())
        p3_b.rules = {"pass_through_nodes": ["Dequantize"], "lowering_map": {}}
        # When it visits n2, it will map n1 to something?
        # Let us just manually mutate n2 inputs before we feed it? No, the pass does:
        # if node_id in dequantize_map: new_inputs[i] = dequantize_map[...]
        # so let us mock that!

        # Dequantize creates a mapping in dequantize_map
        p3_b(graph)

        # Check that 266 was hit: n2 inputs are updated but n2 op_type is the same
        assert graph.nodes["n2"].inputs == ["some_input"]
