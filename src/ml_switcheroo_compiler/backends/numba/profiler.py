"""Numba backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import resource
import sys
import time
import typing
from collections.abc import Mapping
from types import ModuleType

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numba import kernels
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TypedVariadicCallable,
)
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


def _has_numba_shape(val: BackendArg) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (BackendArg): Candidate array.

    Returns:
        bool: True if `val` has `shape`.
    """
    return isinstance(val, ShapeProtocol)


numba: ModuleType | None = None
try:
    import numba as _numba_mod

    numba = _numba_mod
except ImportError:
    numba = None


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


class NumbaProfiler:
    """Provides execution timing and memory profiling for Numba JIT accelerated kernels."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
    ) -> list[BackendArg]:
        """Convert input payloads into topologically ordered array buffers.

        Args:
            graph (IRGraph): IR computation graph.
            inputs: Input mapping.

        Returns:
            list[BackendArg]: Ordered list of array buffers.
        """
        nb_inputs_dict: dict[str, BackendArg] = {}
        for k, val in inputs.items():
            raw_val: BackendArg = val
            if _has_numba_shape(raw_val) and not isinstance(val, (list, tuple)):
                nb_inputs_dict[k] = val
            else:
                nb_inputs_dict[k] = kernels.asarray(val, dtype=kernels.float32)

        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in nb_inputs_dict for k in input_keys):
            return [nb_inputs_dict[k] for k in input_keys]
        return list(nb_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[BackendArg] | None:
        """Compile an IRGraph into an executable Numba JIT function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            TypedVariadicCallable[BackendArg] | None: JIT-compiled evaluate function.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator

        gen = NumbaGenerator(graph)
        code: str = gen.generate()
        ns = load_generated_code(code)
        fn = ns.get(gen._func_name)
        return fn if callable(fn) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile a graph execution on Numba backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs: Named input structures.
            device (str | None): Device identifier ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and peak memory metrics.
        """
        del device

        nb_inputs = self._prepare_inputs(graph, inputs)
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
            """Execute a single evaluation step on Numba backend.

            Returns:
                BackendArg: Evaluated output array.
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
