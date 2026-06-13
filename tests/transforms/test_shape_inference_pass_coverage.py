"""Provides required module functionality."""

from ml_switcheroo_compiler.transforms.passes.shape_inference import shape_inference_pass
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_shape_inference_pass_coverage_brute() -> None:
    """Execute the requested function."""
    g = IRGraph()

    n1 = IRNode(
        id="n1", op_type="Constant", inputs=[], attributes={"value": [1, 2]}, shape_metadata=(2,)
    )
    n2 = IRNode(
        id="n2", op_type="Constant", inputs=[], attributes={"value": [1, 2]}, shape_metadata=None
    )
    n3 = IRNode(id="n3", op_type="Input", inputs=[], attributes={}, shape_metadata=(2, 2))

    n4 = IRNode(id="n4", op_type="Add", inputs=["n3", "n3"], attributes={}, shape_metadata=(2, 2))

    n5 = IRNode(
        id="n5",
        op_type="Reshape",
        inputs=["n3"],
        attributes={"newshape": (4,)},
        shape_metadata=(4,),
    )

    g.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4, "n5": n5}
    shape_inference_pass(g)
