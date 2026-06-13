"""Module docstring."""

from ml_switcheroo_compiler.transforms.passes.dce import dce_pass
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_dce_coverage_brute_loop() -> None:
    """Function docstring."""
    g = IRGraph()
    n2 = IRNode(id="n2", op_type="Input", inputs=["n4"], attributes={}, shape_metadata=None)
    n3 = IRNode(id="n3", op_type="Add", inputs=["n2", "n2"], attributes={}, shape_metadata=None)
    n4 = IRNode(id="n4", op_type="Add", inputs=[], attributes={}, shape_metadata=None)
    g.nodes = {"n2": n2, "n3": n3, "n4": n4}
    g.outputs = ["n3"]

    dce_pass(g)
