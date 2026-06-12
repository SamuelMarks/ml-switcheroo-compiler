"""Unit tests for the shape-aware rewrite transformation.

This module contains test cases to verify that the `shape_aware_rewrite` function
correctly processes logical graphs, specifically handling reshape metadata extraction
and strict cast requirements.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.transforms.rewriter import shape_aware_rewrite


def test_shape_aware_rewrite_reshape() -> None:
    """Tests that a Reshape node with shape metadata is correctly rewritten.

    Verifies that the shape-aware rewriter extracts the `shape_metadata` from
    a Reshape node and populates the `explicit_shape` attribute in the
    rewritten node

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(
        id="n2",
        op_type="Reshape",
        inputs=["n1"],
        shape_metadata=(2, 4),
    )

    rg = shape_aware_rewrite(g)
    assert "explicit_shape" in rg.nodes["n2"].attributes
    assert rg.nodes["n2"].attributes["explicit_shape"] == [2, 4]


def test_shape_aware_rewrite_casts() -> None:
    """Tests that nodes requiring strict casting are correctly identified.

    Verifies that the shape-aware rewriter flags operations like Add with
    the `requires_strict_cast` attribute set to True

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n3"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Input")
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1", "n2"])

    rg = shape_aware_rewrite(g)
    assert rg.nodes["n3"].attributes.get("requires_strict_cast") is True


def test_shape_aware_rewrite_reshape_no_metadata() -> None:
    """Tests that a Reshape node without shape metadata is handled gracefully.

    Verifies that the shape-aware rewriter does not add an `explicit_shape`
    attribute to a Reshape node if its `shape_metadata` is None

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(
        id="n2",
        op_type="Reshape",
        inputs=["n1"],
        shape_metadata=None,
    )
    rg = shape_aware_rewrite(g)
    assert "explicit_shape" not in rg.nodes["n2"].attributes
