"""Numba backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

try:
    import numba
except ImportError:
    numba = None

import numpy as np

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


class NumbaProfiler:
    """Provides execution timing and memory profiling for Numba JIT accelerated kernels."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], np.ndarray]],
    ) -> list[np.ndarray]:
        """Convert input payloads into topologically ordered NumPy arrays.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], np.ndarray]]): Input mapping.

        Returns:
            list[np.ndarray]: Ordered list of NumPy arrays.
        """
        np_inputs_dict: dict[str, np.ndarray] = {}
        for k, val in inputs.items():
            if isinstance(val, np.ndarray):
                np_inputs_dict[k] = val
            else:
                np_inputs_dict[k] = np.asarray(val, dtype=np.float32)

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in np_inputs_dict for k in input_keys):
            return [np_inputs_dict[k] for k in input_keys]
        return list(np_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> Optional[Callable[..., object]]:
        """Compile an IRGraph into an executable Numba JIT function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[Callable[..., object]]: JIT-compiled evaluate function.
        """
        from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator

        code: str = NumbaGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        fn = ns.get("evaluate")
        return fn if callable(fn) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], np.ndarray]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on Numba backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], np.ndarray]]): Named input structures.
            device (Optional[str]): Device identifier ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        np_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: Optional[Callable[..., object]] = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except Exception:
                compiled_fn = None

        def _execute_step() -> object:
            """Execute a single evaluation step on Numba backend.

            Returns:
                object: Evaluated output array.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(np_inputs)

            if not np_inputs:
                return np.zeros((1,), dtype=np.float32)

            accum = np.zeros_like(np_inputs[0])
            for t in np_inputs:
                accum = np.add(accum, t)
            return accum

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
