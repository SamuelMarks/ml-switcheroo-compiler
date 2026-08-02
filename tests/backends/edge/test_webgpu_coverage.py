"""Tests for webgpu backend coverage."""

from ml_switcheroo_compiler.backends.edge.webgpu import WebGPUCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_webgpu_coverage() -> None:
    """Test webgpu generic visit."""
    gen = WebGPUCodeGenerator(IRGraph())
    nodes = [
        LogicalNode(id="n_add", op_type="Add", inputs=["in1", "in2"]),
        LogicalNode(id="n_exp", op_type="Exp", inputs=["in1"]),
        LogicalNode(id="n_min", op_type="Min", inputs=["in1", "in2"]),
        LogicalNode(id="n_neg", op_type="Negative", inputs=["in1"]),
        LogicalNode(id="n_other", op_type="OtherOp", inputs=["in1"]),
    ]
    for n in nodes:
        gen.generic_visit(n, [])
    gen._generate_js_orchestrator("", [], [], 0, 1)


def test_webgpu_more() -> None:
    """Test webgpu more operations."""
    gen = WebGPUCodeGenerator(IRGraph())
    for op in ["Log", "Sqrt", "Abs", "Max", "Subtract", "Multiply", "TrueDivide", "Div", "SomeOtherOp"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen.generic_visit(n, [])
