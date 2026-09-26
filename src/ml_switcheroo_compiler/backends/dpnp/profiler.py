"""DPNP backend profiler for runtime execution and memory benchmarking on Intel SYCL devices."""

from __future__ import annotations

import importlib
import resource
import sys
import time
from typing import Callable

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


def _get_dpnp_peak_memory_mb() -> float:
    """Retrieve peak memory allocation on Intel SYCL device or host process fallback.

    Returns:
        float: Peak memory in megabytes.
    """
    try:
        dpnp_mod = importlib.import_module("dpnp")
        if hasattr(dpnp_mod, "get_device_memory_info"):
            info = dpnp_mod.get_device_memory_info()
            if isinstance(info, (tuple, list)) and len(info) >= 2:
                total, free = info[0], info[1]
                if isinstance(total, (int, float)) and isinstance(free, (int, float)):
                    return float(total - free) / (1024.0 * 1024.0)
    except Exception:
        pass
    return _get_process_memory_mb()


def _sync_dpnp_result(result: object, sycl_queue: object | None = None) -> None:  # noqa: C901
    """Synchronize SYCL execution queue or DPNP array evaluations.

    Args:
        result (object): Evaluated array or collection of arrays.
        sycl_queue (object | None): Optional SYCL execution queue.
    """
    if sycl_queue is not None and hasattr(sycl_queue, "wait"):
        try:
            sycl_queue.wait()
        except Exception:
            pass

    if hasattr(result, "sycl_queue") and hasattr(result.sycl_queue, "wait"):
        try:
            result.sycl_queue.wait()
        except Exception:
            pass
    elif hasattr(result, "wait"):
        try:
            result.wait()
        except Exception:
            pass
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_dpnp_result(item, sycl_queue=sycl_queue)
    elif isinstance(result, dict):
        for val in result.values():
            _sync_dpnp_result(val, sycl_queue=sycl_queue)


class DpnpProfiler:
    """Provides execution timing and peak memory profiling for DPNP on Intel SYCL devices."""

    def __init__(self, sycl_queue: object | None = None) -> None:
        """Initialize DPNP profiler with optional SYCL queue.

        Args:
            sycl_queue (object | None): Optional target SYCL queue.
        """
        self.sycl_queue: object | None = sycl_queue

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | object],
    ) -> list[object]:
        """Convert input payloads into topologically ordered DPNP/NumPy arrays.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, list[float] | list[list[float]] | object]): Input mapping.

        Returns:
            list[object]: Ordered list of input arrays.
        """
        dpnp_inputs: dict[str, object] = {}
        for k, val in inputs.items():
            if type(val).__name__ == "Tensor" and hasattr(val, "data"):
                dpnp_inputs[k] = val.data
            elif type(val).__name__ == "ndarray" or hasattr(val, "ndim"):
                dpnp_inputs[k] = val
            elif isinstance(val, (list, tuple)):
                from ml_switcheroo_compiler.backends.dpnp.types import asarray

                dpnp_inputs[k] = asarray(val)
            else:
                dpnp_inputs[k] = val

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in dpnp_inputs for k in input_keys):
            return [dpnp_inputs[k] for k in input_keys]
        return list(dpnp_inputs.values())

    def _compile_runner(self, graph: IRGraph, device: str = "auto") -> Callable[..., object] | None:
        """Compile computational graph into a DPNP executable runner.

        Args:
            graph (IRGraph): Target computation graph.
            device (str): SYCL target device.

        Returns:
            Callable[..., object] | None: Executable runner if compiled.
        """
        from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator

        gen = DPNPGenerator(graph, device=device, sycl_queue=self.sycl_queue)
        runner = gen._compile_aot_impl(graph)
        return runner if callable(runner) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | object],
        device: str = "auto",
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, list[float] | float]:
        """Profile graph execution on DPNP backend targeting Intel SYCL.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, list[float] | list[list[float]] | object]): Named inputs.
            device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, list[float] | float]: Latency and peak memory metrics.
        """
        prepared_inputs = self._prepare_inputs(graph, inputs)
        runner = self._compile_runner(graph, device=device)
        if runner is None:
            return {
                "latencies_ms": [],
                "mean_ms": 0.0,
                "median_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "peak_memory_mb": _get_dpnp_peak_memory_mb(),
            }

        # Warmup executions
        for _ in range(max(1, warmup_iters)):
            out = runner(*prepared_inputs)
            _sync_dpnp_result(out, sycl_queue=self.sycl_queue)

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            t0 = time.perf_counter()
            out = runner(*prepared_inputs)
            _sync_dpnp_result(out, sycl_queue=self.sycl_queue)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = sum(latencies) / len(latencies) if latencies else 0.0
        sorted_lats = sorted(latencies)
        median_ms = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0.0
        min_ms = min(latencies) if latencies else 0.0
        max_ms = max(latencies) if latencies else 0.0
        peak_mb = _get_dpnp_peak_memory_mb()

        return {
            "latencies_ms": latencies,
            "mean_ms": float(mean_ms),
            "median_ms": float(median_ms),
            "min_ms": float(min_ms),
            "max_ms": float(max_ms),
            "peak_memory_mb": float(peak_mb),
        }
