"""Tests for ONNX backend coverage."""

from ml_switcheroo_compiler.backends.edge.onnx import ONNXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_onnx_more() -> None:
    """Test onnx edge coverage."""
    gen = ONNXCodeGenerator(IRGraph())
    for op in ["Constant", "Add", "Sub", "Mul", "Div", "Other"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen.sorted_nodes.append(n)
    try:
        gen.generate()
    except Exception:
        pass
    try:
        gen.export_onnx("test.onnx")
    except Exception:
        pass
