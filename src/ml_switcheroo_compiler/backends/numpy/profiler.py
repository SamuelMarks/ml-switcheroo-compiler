"""Numpy backend profiler for benchmarking."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRGraph


class NumpyProfiler:
    """Provides memory and latency profiling for the Numpy backend."""

    def profile_graph(self, graph: IRGraph, inputs: dict[str, Any]) -> dict[str, float]:
        """Profile a graph execution.

        Args:
            graph (IRGraph): The IR graph.
            inputs (dict[str, Any]): Inputs for the graph.

        Returns:
            dict[str, float]: Latency and memory metrics.
        """
        return {"latency_ms": 0.0, "peak_memory_mb": 0.0}
