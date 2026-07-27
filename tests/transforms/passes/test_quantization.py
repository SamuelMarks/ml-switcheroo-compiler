"""Tests for quantization pass."""

from ml_switcheroo_compiler.core.dataset import Dataset
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ir.core import IRNode, LogicalGraph
from ml_switcheroo_compiler.transforms.passes.quantization import PTQPass, QuantizationConfig


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
    graph = LogicalGraph(name="test", nodes=nodes, outputs=["add"])

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
    graph = LogicalGraph(name="test", nodes=nodes, outputs=["add"])

    ptq = PTQPass(config, dataset)
    optimized_graph = ptq(graph)
    assert "ptq_target_dtype" not in optimized_graph.nodes["add"].attributes
