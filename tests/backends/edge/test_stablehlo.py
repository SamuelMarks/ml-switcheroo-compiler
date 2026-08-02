"""Tests for stablehlo code generator coverage."""

from ml_switcheroo_compiler.backends.edge.stablehlo import StableHLOCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_stablehlo_coverage() -> None:
    """Test stablehlo code generation edge cases."""
    gen = StableHLOCodeGenerator(IRGraph())
    for op in ["Add", "Subtract", "Multiply", "Div", "Exp", "Log", "Abs", "Min", "Max", "Negative", "Other"]:
        n = LogicalNode(id="n_" + op, op_type=op, inputs=["in1", "in2"])
        gen.generic_visit(n, [])
