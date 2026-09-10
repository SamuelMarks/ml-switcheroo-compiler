"""Numpy backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Any, Optional, Union

import numpy as np

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


class NumpyProfiler:
    """Provides memory and latency profiling for the Numpy backend."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], np.ndarray]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on NumPy backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], np.ndarray]]): Named input arrays.
            device (Optional[str]): Device identifier ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and memory metrics.
        """
        del device

        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        np_inputs_dict: dict[str, np.ndarray] = {}
        for k, val in inputs.items():
            if isinstance(val, np.ndarray):
                np_inputs_dict[k] = val
            else:
                np_inputs_dict[k] = np.asarray(val, dtype=np.float32)

        np_input_list: list[np.ndarray] = list(np_inputs_dict.values())

        def _execute_step() -> Any:
            """Execute a single evaluation step on NumPy backend.

            Returns:
                Any: Single or dictionary of output arrays.
            """
            if has_ops:
                from ml_switcheroo_compiler.backends.numpy.generator import NumpyGenerator
                from ml_switcheroo_compiler.interpreter.evaluator import evaluate_graph

                return evaluate_graph(graph, np_inputs_dict, backend=NumpyGenerator)
            if not np_input_list:
                return np.zeros((1,), dtype=np.float32)
            accum = np.zeros_like(np_input_list[0])
            for t in np_input_list:
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

        mean_lat: float = float(sum(latencies) / len(latencies)) if latencies else 0.0
        peak_mem: float = _get_process_memory_mb()

        return {
            "latencies": latencies,
            "latency_ms": mean_lat,
            "peak_memory_mb": peak_mem,
            "warmup_iterations": max(1, warmup_iters),
        }
