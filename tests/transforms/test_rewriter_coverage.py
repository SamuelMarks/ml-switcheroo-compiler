"""Tests for test_rewriter_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.transforms.rewriter as rewriter_mod


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
