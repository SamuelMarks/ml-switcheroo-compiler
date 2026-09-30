"""Tests for shape-aware rewriting pass in transforms.rewriter."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.transforms.rewriter import shape_aware_rewrite


def test_shape_aware_rewrite() -> None:
    """Test shape_aware_rewrite converts Reshape shapes and flags arithmetic operations.

    Returns:
        None.
    """
    graph = LogicalGraph(name="test_graph", outputs=["add"])
    node = LogicalNode(id="reshape", op_type="Reshape", inputs=[], shape_metadata=(10,), attributes={})
    graph.nodes["reshape"] = node

    node_no_shape = LogicalNode(id="reshape2", op_type="Reshape", inputs=[], shape_metadata=None, attributes={})
    graph.nodes["reshape2"] = node_no_shape

    node2 = LogicalNode(id="add", op_type="Add", inputs=[], shape_metadata=None, attributes={})
    graph.nodes["add"] = node2

    for op in ["Sub", "Mul", "Div", "MatMul"]:
        graph.nodes[op.lower()] = LogicalNode(id=op.lower(), op_type=op, inputs=[], attributes={})

    new_graph = shape_aware_rewrite(graph)
    assert new_graph.nodes["reshape"].attributes["explicit_shape"] == [10]
    assert "explicit_shape" not in new_graph.nodes["reshape2"].attributes
    assert new_graph.nodes["add"].attributes["requires_strict_cast"] is True
    for op in ["Sub", "Mul", "Div", "MatMul"]:
        assert new_graph.nodes[op.lower()].attributes["requires_strict_cast"] is True
