"""Module docstring."""

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def get_graph(op_type: str) -> IRGraph:
    """Docstring."""
    g = IRGraph()
    n1 = IRNode(
        id="n1",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0]},
        shape_metadata=None,
    )
    n2 = IRNode(id="n2", op_type="Input", inputs=[], attributes={}, shape_metadata=(2,))
    n3 = IRNode(
        id="n3",
        op_type=op_type,
        inputs=["n1", "n2"],
        attributes={"axis": 0},
        shape_metadata=None,
    )
    for n in [n1, n2, n3]:
        g.nodes[n.id] = n
    g.inputs = ["n2"]
    g.outputs = ["n3"]
    return g


def test_backends_brute_coverage_specifics() -> None:
    pass
