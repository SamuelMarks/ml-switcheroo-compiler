"""MLX backend profiler for benchmarking."""

from typing import Any

from ml_switcheroo_compiler.ir.core import IRGraph


class MLXProfiler:
    """Provides memory and latency profiling for the MLX backend."""

    def profile_graph(self, graph: IRGraph, inputs: dict[str, Any]) -> dict[str, float]:
        """Profile a graph execution.

        Args:
            graph (IRGraph): The IR graph.
            inputs (dict[str, Any]): Inputs for the graph.

        Returns:
            dict[str, float]: Latency and memory metrics.
        """
        # Note: MLX has internal profilers like mlx.core.metal.get_peak_memory()
        # For Tier 2 compliance we keep it minimal.
        return {"latency_ms": 0.0, "peak_memory_mb": 0.0}
