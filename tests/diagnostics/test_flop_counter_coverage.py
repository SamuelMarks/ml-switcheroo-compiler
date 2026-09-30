"""Tests for test_flop_counter_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.diagnostics.flop_counter as flop_mod


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


def test_flop_counter_exhaustive() -> None:
    """Verify flop counter across MatMul, valid shapes, invalid shapes, and unknown ops."""
    graph = LogicalGraph(name="test_graph", outputs=["out"])

    # 1. MatMul node
    n_matmul = LogicalNode("n0", "MatMul", inputs=["a", "b"])
    graph.nodes["n0"] = n_matmul

    # 2. Node with valid shape_metadata
    n_shape = LogicalNode("n1", "Relu", inputs=["n0"])
    n_shape.shape_metadata = (4, 8, 16)
    graph.nodes["n1"] = n_shape

    # 3. Node with invalid shape_metadata (raises TypeError)
    n_bad_shape = LogicalNode("n2", "Relu", inputs=["n1"])
    n_bad_shape.shape_metadata = _DummyBadShape()
    graph.nodes["n2"] = n_bad_shape

    # 4. Node without shape_metadata and op_type != 'Foo'
    n_other = LogicalNode("n3", "CustomOp", inputs=["n2"])
    graph.nodes["n3"] = n_other

    # 5. Node with op_type == 'Foo' without shape_metadata
    n_foo = LogicalNode("n4", "Foo", inputs=["n3"])
    graph.nodes["n4"] = n_foo

    total = flop_mod.estimate_flops(graph)
    # 100 (MatMul) + 512 (4*8*16) + 1 (bad shape) + 1 (CustomOp) + 0 (Foo) = 614
    assert total == 614
