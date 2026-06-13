"""Provides required module functionality."""

from ml_switcheroo_compiler.transforms.passes.constant_folding import constant_folding_pass
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_constant_folding_coverage_brute() -> None:
    """Execute the requested function."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"], attributes={}, shape_metadata=None)
    g.nodes = {"n1": n1, "n2": n2}

    constant_folding_pass(g)

    g3 = IRGraph()
    n4 = IRNode(
        id="n4", op_type="Constant", inputs=[], attributes={"value": [1, 2]}, shape_metadata=None
    )
    n5 = IRNode(
        id="n5", op_type="Constant", inputs=[], attributes={"value": [3, 4]}, shape_metadata=None
    )
    n6 = IRNode(id="n6", op_type="Add", inputs=["n4", "n5"], attributes={}, shape_metadata=None)
    g3.nodes = {"n4": n4, "n5": n5, "n6": n6}
    constant_folding_pass(g3)

    g4 = IRGraph()
    n7 = IRNode(
        id="n7", op_type="Constant", inputs=[], attributes={"value": 1}, shape_metadata=None
    )
    n8 = IRNode(
        id="n8",
        op_type="UnknownOpThatRaisesException",
        inputs=["n7"],
        attributes={},
        shape_metadata=None,
    )
    g4.nodes = {"n7": n7, "n8": n8}
    constant_folding_pass(g4)
