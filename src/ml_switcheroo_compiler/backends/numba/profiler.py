"""Numba backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import resource
import sys
import time
from typing import Callable

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.backends.numba import kernels
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
        inputs: dict[str, list[float] | list[list[float]] | EagerValue],
    ) -> list[EagerValue]:
        """Convert input payloads into topologically ordered array buffers.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, list[float] | list[list[float]] | EagerValue]): Input mapping.

        Returns:
            list[EagerValue]: Ordered list of array buffers.
        """
        nb_inputs_dict: dict[str, EagerValue] = {}
        for k, val in inputs.items():
            if hasattr(val, "shape") and not isinstance(val, (list, tuple)):
                nb_inputs_dict[k] = val
            else:
                nb_inputs_dict[k] = kernels.asarray(val, dtype=kernels.float32)

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in nb_inputs_dict for k in input_keys):
            return [nb_inputs_dict[k] for k in input_keys]
        return list(nb_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> Callable[..., EagerValue] | None:
        """Compile an IRGraph into an executable Numba JIT function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Callable[..., EagerValue] | None: JIT-compiled evaluate function.
        """
        from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator

        gen = NumbaGenerator(graph)
        code: str = gen.generate()
        ns: dict[str, EagerValue] = {}
        exec(code, ns)
        fn = ns.get(gen._func_name)
        return fn if callable(fn) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | EagerValue],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, list[float] | float]:
        """Profile a graph execution on Numba backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, list[float] | list[list[float]] | EagerValue]): Named input structures.
            device (str | None): Device identifier ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, list[float] | float]: Latency and peak memory metrics.
        """
        del device

        nb_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: Callable[..., EagerValue] | None = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except Exception:
                compiled_fn = None

        def _execute_step() -> EagerValue:
            """Execute a single evaluation step on Numba backend.

            Returns:
                EagerValue: Evaluated output array.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(tuple(nb_inputs))

            if not nb_inputs:
                return kernels.zeros((1,), dtype=kernels.float32)

            accum = kernels.zeros_like(nb_inputs[0])
            for t in nb_inputs:
                accum = kernels.add(accum, t)
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
            "p50_ms": p50_lat,
            "p95_ms": p95_lat,
            "p99_ms": p99_lat,
            "peak_memory_mb": peak_mem,
            "warmup_iterations": max(1, warmup_iters),
        }
