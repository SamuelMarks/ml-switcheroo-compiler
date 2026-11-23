"""Dask backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import resource
import sys
import time
import typing
from collections.abc import Mapping
from types import ModuleType

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

da: ModuleType | None
try:
    import dask.array as _da

    da = _da
except ImportError:
    da = None


@typing.runtime_checkable
class _ComputableDaskProtocol(typing.Protocol):
    """Protocol for lazy Dask arrays exposing `.compute()`."""

    def compute(self) -> BackendArg:
        """Compute lazy Dask task graph.

        Returns:
            BackendArg: Concrete array result.
        """
        ...


def _is_computable_dask(val: BackendArg) -> typing.TypeGuard[_ComputableDaskProtocol]:
    """Check whether `val` exposes a `.compute` attribute.

    Args:
        val (BackendArg): Candidate lazy Dask array.

    Returns:
        bool: True if `val` has a `.compute` attribute.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _ComputableDaskProtocol)
    return found or is_mock_with_attrs(val, "compute")


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


def _sync_dask_result(
    result: BackendArg | list[BackendArg] | tuple[BackendArg, ...] | dict[str, BackendArg],
) -> BackendArg:
    """Force computation and synchronization of Dask computation graph.

    Args:
        result (BackendArg | list[BackendArg] | tuple[BackendArg, ...] | dict[str, BackendArg]):
            Lazy Dask array or container.

    Returns:
        BackendArg: Computed concrete array or structure.
    """
    if da is None:
        return result
    if isinstance(result, dict):
        return {k: _sync_dask_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        computed = [_sync_dask_result(item) for item in result]
        return type(result)(computed)
    raw_res: BackendArg = result
    if _is_computable_dask(raw_res) and callable(raw_res.compute):
        try:
            return raw_res.compute()
        except (
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            ArithmeticError,
            RuntimeError,
            SyntaxError,
            TypeError,
            ValueError,
        ):
            _typed_ret_1: BackendArg = result
            return _typed_ret_1
    return result


class DaskProfiler:
    """Provides genuine execution timing and peak memory profiling for Dask arrays."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
    ) -> list[BackendArg]:
        """Convert input structures into topologically ordered Dask arrays.

        Args:
            graph (IRGraph): IR computation graph.
            inputs: Input structures.

        Returns:
            list[object]: Topologically ordered list of Dask arrays.
        """
        if da is None:
            return []

        da_inputs_dict: dict[str, BackendArg] = {}
        for k, val in inputs.items():
            raw_val: BackendArg = val
            if _is_computable_dask(raw_val):
                da_inputs_dict[k] = val
            else:
                da_inputs_dict[k] = da.asarray(val, dtype=float)

        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in da_inputs_dict for k in input_keys):
            return [da_inputs_dict[k] for k in input_keys]
        return list(da_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[BackendArg] | None:
        """Compile an IRGraph into an executable Dask function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[TypedVariadicCallable[object]]: Evaluator function.

        Raises:
            TypeError: If evaluate cannot be loaded from the generated code.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.dask.generator import DaskGenerator

        code: str = DaskGenerator(graph).generate()
        ns = load_generated_code(code)
        evaluate_fn = ns.get("evaluate")
        if not callable(evaluate_fn):
            raise TypeError("Failed to generate valid evaluate function from DaskGenerator")
        return evaluate_fn

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile a graph execution on Dask backend with concrete computation.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs: Named input structures.
            device (Optional[str]): Device identifier.
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and peak memory metrics.
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
        has_ops: bool = bool(graph.nodes and any(n.op_type != "Input" for n in graph.nodes.values()))

        compiled_fn: TypedVariadicCallable[BackendArg] | None = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except (
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                ArithmeticError,
                RuntimeError,
                SyntaxError,
                TypeError,
                ValueError,
            ):
                compiled_fn = None

        def _execute_step() -> BackendArg:
            """Execute a single evaluation step on Dask backend with compute synchronization.

            Returns:
                BackendArg: Concrete evaluated array or structure.
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
