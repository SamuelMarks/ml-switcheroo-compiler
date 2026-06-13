"""Module docstring."""

from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_dce_coverage_brute2() -> None:
    """Function docstring."""
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Input", inputs=[], attributes={}, shape_metadata=None)
    g.nodes = {"n1": n1}
    g.outputs = ["n1"]

    dce_pass(g)
    assert "n1" in g.nodes
