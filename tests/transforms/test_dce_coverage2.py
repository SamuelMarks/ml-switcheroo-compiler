"""Provides required module functionality."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from ml_switcheroo_compiler.transforms.passes.dce import dce_pass


def test_dce_coverage_brute2() -> None:
    """Execute the requested function."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    g.nodes = {"n1": n1}
    g.outputs = ["n1"]

    dce_pass(g)
    assert "n1" in g.nodes
