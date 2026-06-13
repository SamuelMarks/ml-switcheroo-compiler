"""Unit tests for the Dead Code Elimination (DCE) transformation pass.

This module contains test cases to verify that the DCE pass correctly identifies and
removes unused nodes from a logical graph representation.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.passes.dce import dce_pass


def test_dce() -> None:
    """Tests that the DCE pass removes disconnected nodes and their unused dependencies.

    This test constructs a logical graph where only one node is marked as an output
    It verifies that other independent nodes, as well as nodes that depend on them
    but do not contribute to the output, are successfully pruned from the graph

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Add", inputs=[])
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=[])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n1"])
    dce_pass(g)
    assert "n1" not in g.nodes
    assert "n3" not in g.nodes
    assert "n2" in g.nodes


def test_dce_chained() -> None:
    """Tests that the DCE pass removes a chain of unused dependent nodes.

    This test constructs a logical graph where an output node has a chain of
    descendant nodes depending on it, but those descendants do not contribute
    to any graph outputs. It verifies that the entire unused chain is pruned,
    leaving only the required output node

    Returns:
    None
    """
    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    g.nodes["n3"] = LogicalNode(id="n3", op_type="Add", inputs=["n2"])
    g.nodes["n4"] = LogicalNode(id="n4", op_type="Add", inputs=["n3"])
    dce_pass(g)
    assert "n2" not in g.nodes
    assert "n3" not in g.nodes
    assert "n4" not in g.nodes
    assert "n1" in g.nodes
