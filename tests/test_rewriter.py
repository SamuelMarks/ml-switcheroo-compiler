"""Module docstring."""

from ml_switcheroo_ir import LogicalGraph, LogicalNode
from ml_switcheroo_compiler.rewriter import shape_aware_rewrite


def test_shape_aware_rewrite_reshape():
    """Docstring."""
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(
        id="n2", op_type="Reshape", inputs=["n1"], shape_metadata=(2, 4)
    )

    rg = shape_aware_rewrite(g)
    assert "explicit_shape" in rg.nodes["n2"].attributes
    assert rg.nodes["n2"].attributes["explicit_shape"] == [2, 4]


def test_shape_aware_rewrite_casts():
    """Docstring."""
    g = LogicalGraph(outputs=["n3"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Input")
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1", "n2"])

    rg = shape_aware_rewrite(g)
    assert rg.nodes["n3"].attributes.get("requires_strict_cast") is True
