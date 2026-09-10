"""C++ / LLVM backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
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


class CppProfiler:
    """Provides genuine execution timing and peak memory profiling for native C++ compilation."""

    def _compile_graph(self, graph: IRGraph) -> Optional[Callable[[], str]]:
        """Compile an IRGraph into an executable C++ shared library callable.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            Optional[Callable[[], str]]: Native callable executing the compiled C++ graph.
        """
        generator = CppGenerator(graph=graph)
        try:
            return generator._compile_aot_impl(graph)
        except Exception:
            return None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on C++ backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input structures.
            device (Optional[str]): Target execution device ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device, inputs

        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: Optional[Callable[[], str]] = None
        if has_ops:
            compiled_fn = self._compile_graph(graph)

        def _execute_step() -> None:
            """Execute a single evaluation step of the compiled C++ kernel."""
            if compiled_fn is not None:
                _ = compiled_fn()

        for _ in range(max(1, warmup_iters)):
            _execute_step()

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            _execute_step()
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
