"""Tests for test_memory_profiler_coverage."""

from __future__ import annotations

from ml_switcheroo_ir import LogicalGraph, LogicalNode


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


def test_memory_profiler_exhaustive() -> None:
    """Verify memory profiler across valid shapes, bad shapes, and missing metadata."""
    from ml_switcheroo_compiler.diagnostics.memory_profiler import memory_profiler

    graph = LogicalGraph(name="test_mem_graph", outputs=["out"])

    # 1. Valid shape metadata: 2 * 3 * 4 = 24 elements * 4 bytes = 96
    n0 = LogicalNode("n0", "Add")
    n0.shape_metadata = (2, 3, 4)
    graph.nodes["n0"] = n0

    # 2. Bad shape metadata (raises TypeError): 4 bytes fallback
    n1 = LogicalNode("n1", "Sub")
    n1.shape_metadata = _DummyBadShape()
    graph.nodes["n1"] = n1

    # 3. Missing shape metadata: 4 bytes fallback
    n2 = LogicalNode("n2", "Mul")
    graph.nodes["n2"] = n2

    total_mem = memory_profiler(graph)
    assert total_mem == 96 + 4 + 4
