"""Tests for wasm backend coverage."""

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_wasm_coverage() -> None:
    """Test wasm code generator coverage."""
    gen = WasmCodeGenerator(IRGraph())
    gen.is_simd = True
    ops = ["Subtract", "Multiply", "Div", "Sqrt", "Abs", "Min", "Max", "Exp", "Negative", "OtherOp"]
    for op in ops:
        n = LogicalNode(id=f"n_{op.lower()}", op_type=op, inputs=["in1", "in2"])
        gen._visit_simd(n, f"n_{op.lower()}", op, f"v_n_{op.lower()}")


def test_wasm_more() -> None:
    """Test wasm more operations."""
    gen = WasmCodeGenerator(IRGraph())
    gen.is_simd = True
    for op in ["Log", "Add", "TrueDivide"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen._visit_simd(n, "id", op, "var")
    gen.is_simd = False
    for op in ["Log", "Sqrt", "Abs", "Min", "Max", "Negative", "Neg", "Add", "Subtract", "Multiply", "TrueDivide", "Div", "Other"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen._visit_scalar(n, "id", op, "var")
