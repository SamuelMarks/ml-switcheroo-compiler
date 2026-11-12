"""Tests for test_flop_counter_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode

import ml_switcheroo_compiler.diagnostics.flop_counter as flop_mod
from tests.testing_utils import ensure_eager_tensor, ensure_mutable


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


def test_flop_counter_exhaustive() -> None:
    """Verify flop counter across MatMul, valid shapes, invalid shapes, and unknown ops."""
    graph = LogicalGraph(name="test_graph", outputs=["out"])

    # 1. MatMul node with input shapes (4, 8) and (8, 5) -> 2 * 4 * 5 * 8 = 320 FLOPs
    graph.nodes["a"] = LogicalNode("a", "Input", shape_metadata=(4, 8))
    graph.nodes["b"] = LogicalNode("b", "Input", shape_metadata=(8, 5))
    n_matmul = LogicalNode("n0", "MatMul", inputs=["a", "b"], shape_metadata=(4, 5))
    graph.nodes["n0"] = n_matmul

    # 2. Node with valid shape_metadata
    n_shape = LogicalNode("n1", "Relu", inputs=["n0"])
    n_shape.shape_metadata = (4, 8, 16)
    graph.nodes["n1"] = n_shape

    # 3. Node with invalid shape_metadata (raises TypeError)
    n_bad_shape = LogicalNode("n2", "Relu", inputs=["n1"])
    ensure_mutable(n_bad_shape).shape_metadata = ensure_eager_tensor(_DummyBadShape())
    graph.nodes["n2"] = n_bad_shape

    # 4. Node without shape_metadata
    n_other = LogicalNode("n3", "CustomOp", inputs=["n2"])
    graph.nodes["n3"] = n_other

    # 5. Zero-flop node (Identity) without shape_metadata
    n_id = LogicalNode("n4", "Identity", inputs=["n3"])
    graph.nodes["n4"] = n_id

    total = flop_mod.estimate_flops(graph)
    # 0 (Inputs a, b) + 320 (MatMul) + 512 (4*8*16) + 1 (bad shape) + 1 (CustomOp) + 0 (Identity) = 834
    assert total == 834
