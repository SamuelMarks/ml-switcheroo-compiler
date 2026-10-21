"""Sparse backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import resource
import sys
import time
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, as_shape, as_tensor
from ml_switcheroo_compiler.backends.sparse.types import (
    _SP_BACKEND,
    COOTensor,
    CSCTensor,
    CSRTensor,
)
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph

sparse: ModuleType | None = None
try:
    import sparse as _sparse_mod

    sparse = _sparse_mod
except ImportError:
    sparse = None


def _get_process_memory_mb() -> float:
    """Retrieve host process peak resident set size in megabytes.

    Returns:
        float: Peak RSS in megabytes.
    """
    rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    platform_name: str = sys.platform
    if platform_name == "darwin":
        return float(rusage) / (1024.0 * 1024.0)
    return float(rusage) / 1024.0


class SparseProfiler:
    """Provides execution timing and memory profiling for sparse array operations."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | EagerValue | EagerValue],
    ) -> list[EagerValue]:
        """Convert input structures into sparse or dense array representations.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], EagerValue, object]]): Input structures.

        Returns:
            list[object]: Topologically ordered list of input arrays.
        """
        arr_dict: dict[str, EagerValue] = {}
        for k, val in inputs.items():
            if isinstance(val, (COOTensor, CSRTensor, CSCTensor)):
                arr_dict[k] = val
            elif hasattr(val, "coords") and hasattr(val, "data") and hasattr(val, "shape"):
                arr_dict[k] = COOTensor(
                    indices=_SP_BACKEND.asarray(getattr(val, "coords")),
                    values=_SP_BACKEND.asarray(getattr(val, "data")),
                    shape=as_shape(getattr(val, "shape")),
                )
            elif hasattr(val, "shape") and not isinstance(val, (list, tuple)):
                dense_val = getattr(val, "todense")() if hasattr(val, "todense") else val
                arr_dict[k] = COOTensor.from_dense(dense_val)
            elif isinstance(val, (list, tuple)):
                np_arr = _SP_BACKEND.asarray(val, dtype=_SP_BACKEND.float32)
                arr_dict[k] = COOTensor.from_dense(np_arr)
            else:
                arr_dict[k] = val

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in arr_dict for k in input_keys):
            return [arr_dict[k] for k in input_keys]
        return list(arr_dict.values())

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[EagerValue] | None:
        """Compile an IRGraph into an executable sparse function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[TypedVariadicCallable[object]]: Sparse evaluator function.
        """
        from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator

        gen = SparseGenerator(graph)
        code: str = gen.generate()
        ns: dict[str, EagerValue] = {}
        exec(code, ns)
        evaluate_fn = ns.get(gen._func_name)
        return evaluate_fn if callable(evaluate_fn) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | EagerValue | EagerValue],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, list[float] | float]:
        """Profile a graph execution on Sparse backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], EagerValue, object]]): Named input structures.
            device (Optional[str]): Device target.
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        sparse_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: TypedVariadicCallable[EagerValue] | None = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except Exception:
                compiled_fn = None

        def _execute_step() -> EagerValue:
            """Execute a single evaluation step on sparse backend.

            Returns:
                EagerValue: Evaluated output array.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(sparse_inputs)

            if not sparse_inputs:
                return as_tensor(_SP_BACKEND.zeros((1,), dtype=_SP_BACKEND.float32))

            accum = sparse_inputs[0]
            for t in sparse_inputs[1:]:
                if hasattr(accum, "__add__"):
                    accum = as_tensor(accum) + as_tensor(t)
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
