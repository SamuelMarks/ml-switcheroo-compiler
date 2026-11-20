"""Behavioral unit tests for rewriter."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.rewriter as rewriter_mod
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


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.

        Returns:
            int: Computed output value from `__next__`.
        """
        raise TypeError("Not iterable dimension")


def test_transforms_rewriter_exhaustive() -> None:
    """Verify shape aware rewrite for Reshape and arithmetic operations."""
    graph = LogicalGraph(name="model", outputs=["out"])

    # 1. Reshape with shape_metadata
    n_reshape = LogicalNode("r0", "Reshape", inputs=["in"])
    n_reshape.shape_metadata = (1, 64)
    graph.nodes["r0"] = n_reshape

    # 2. Arithmetic operations (requires strict cast)
    n_add = LogicalNode("a0", "Add", inputs=["r0", "r0"])
    graph.nodes["a0"] = n_add

    # 3. Passthrough node
    n_relu = LogicalNode("out", "Relu", inputs=["a0"])
    graph.nodes["out"] = n_relu

    rewritten = rewriter_mod.shape_aware_rewrite(graph)
    assert rewritten.name == "model_rewritten"
    assert rewritten.nodes["r0"].attributes["explicit_shape"] == [1, 64]
    assert rewritten.nodes["a0"].attributes["requires_strict_cast"] is True
    assert "requires_strict_cast" not in rewritten.nodes["out"].attributes
