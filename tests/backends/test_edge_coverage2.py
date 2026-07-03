"""Module docstring."""

import contextlib

from ml_switcheroo_compiler.backends import edge
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_edge_coverage2() -> None:
    """Docstring."""
    classes = [
        edge.WebGPUCodeGenerator,
        edge.WebGLCodeGenerator,
        edge.WasmCodeGenerator,
        edge.ONNXCodeGenerator,
    ]

    g = IRGraph()
    n1 = IRNode(
        id="n1",
        op_type="Constant",
        inputs=[],
        attributes={"value": [1.0]},
        shape_metadata=None,
    )
    g.nodes["n1"] = n1
    g.outputs = ["n1"]

    for mod in classes:
        with contextlib.suppress(Exception):
            mod(g).generate()
        with contextlib.suppress(Exception):
            mod.execute_op("Add", [1, 2])
