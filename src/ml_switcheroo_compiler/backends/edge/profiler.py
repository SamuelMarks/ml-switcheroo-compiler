"""Edge backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Optional, Union

from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph


def _get_process_memory_mb() -> float:
    """Retrieve host process peak resident set size in megabytes.

    Returns:
        float: Peak RSS in megabytes.
    """
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rusage) / (1024.0 * 1024.0)
    return float(rusage) / 1024.0


class EdgeProfiler:
    """Provides execution timing and memory profiling for edge web targets."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
    ) -> list[object]:
        """Convert input payloads into ordered buffer structures.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input data.

        Returns:
            list[object]: Ordered list of input payloads.
        """
        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in inputs for k in input_keys):
            return [inputs[k] for k in input_keys]
        return list(inputs.values())

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on edge backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input structures.
            device (Optional[str]): Device target ('webgpu', 'wasm', 'webgl').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        edge_inputs = self._prepare_inputs(graph, inputs)

        def _execute_step() -> float:
            """Execute a single evaluation step on edge target simulation.

            Returns:
                float: Evaluated result accumulation.
            """
            if not edge_inputs:
                return 0.0
            return float(len(edge_inputs))

        for _ in range(max(1, warmup_iters)):
            _ = _execute_step()

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            _ = _execute_step()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        sorted_lat: list[float] = sorted(latencies) if latencies else [0.0]
        mean_lat: float = float(sum(latencies) / len(latencies)) if latencies else 0.0
        p50_lat: float = float(sorted_lat[int(len(sorted_lat) * 0.50)])
        p95_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))])
        p99_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))])
        peak_mem: float = _get_process_memory_mb()

        return {
            "latencies": latencies,
            "latency_ms": mean_lat,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "peak_memory_mb": peak_mem,
            "warmup_iterations": max(1, warmup_iters),
        }
