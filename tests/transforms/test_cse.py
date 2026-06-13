"""Unit tests for the Common Subexpression Elimination (CSE) optimization pass on logical.

graphs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.passes.cse import cse_pass


def test_cse() -> None:
    """Tests that the CSE pass successfully merges identical nodes.

    This test constructs a logical graph with two identical 'Relu' nodes
    sharing the same input. It verifies that after running the CSE pass,
    one of the duplicate nodes is eliminated and its consumers are updated
    to reference the remaining node

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n3", "n4"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Relu", inputs=["in"])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Relu", inputs=["in"])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1", "n2"])
    g.nodes["n4"] = LogicalNode(id="n4", op_type="Add", inputs=["n2", "n1"])
    cse_pass(g)
    assert "n2" not in g.nodes
    assert g.nodes["n3"].inputs == ["n1", "n1"]


def test_cse_different_attributes() -> None:
    """Tests that the CSE pass does not merge nodes with different attributes.

    This test constructs a logical graph with two 'Transpose' nodes that
    have the same input and operation type but different 'axes' attributes
    It verifies that the CSE pass correctly identifies them as distinct
    and does not eliminate either node

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1", "n2"])
    g.nodes["in"] = LogicalNode(id="in", op_type="Input")
    g.nodes["n1"] = LogicalNode(
        id="n1",
        op_type="Transpose",
        inputs=["in"],
        attributes={"axes": [1, 0]},
    )
    g.nodes["n2"] = LogicalNode(
        id="n2",
        op_type="Transpose",
        inputs=["in"],
        attributes={"axes": [0, 1]},
    )
    cse_pass(g)
    assert "n1" in g.nodes
    assert "n2" in g.nodes
