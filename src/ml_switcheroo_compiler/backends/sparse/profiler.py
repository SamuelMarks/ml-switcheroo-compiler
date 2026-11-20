"""Sparse backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import resource
import sys
import time
import typing
from collections.abc import Mapping
from types import ModuleType

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, coerce_shape
from ml_switcheroo_compiler.backends.sparse.types import (
    _SP_BACKEND,
    COOTensor,
    CSCTensor,
    CSRTensor,
)
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import (
    ShapeProtocol,
    TypedVariadicCallable,
)
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


@typing.runtime_checkable
class _SparseCoordsProtocol(typing.Protocol):
    """Protocol for external COO sparse arrays exposing `coords`, `data`, and `shape`."""

    coords: BackendArg
    data: BackendArg
    shape: tuple[int, ...]


@typing.runtime_checkable
class _TodenseProtocol(typing.Protocol):
    """Protocol for sparse arrays exposing `.todense()`."""

    def todense(self) -> BackendArg:
        """Convert sparse array to dense representation.

        Returns:
            BackendArg: Dense array.
        """
        ...


@typing.runtime_checkable
class _AddableOperandProtocol(typing.Protocol):
    """Protocol for operands supporting `+`."""

    def __add__(self, other: BackendArg) -> BackendArg:
        """Add two operands.

        Args:
            other (BackendArg): Right-hand operand.

        Returns:
            BackendArg: Sum result.
        """
        ...


def _is_sparse_coords(val: BackendArg) -> typing.TypeGuard[_SparseCoordsProtocol]:
    """Check whether `val` exposes `coords`, `data`, and `shape`.

    Args:
        val (BackendArg): Candidate sparse array.

    Returns:
        bool: True if `val` has `coords`, `data`, and `shape`.
    """
    return isinstance(val, _SparseCoordsProtocol)


def _is_shape_holder(val: BackendArg) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` exposes a `shape` attribute.

    Args:
        val (BackendArg): Candidate array.

    Returns:
        bool: True if `val` has a `shape` attribute.
    """
    return isinstance(val, ShapeProtocol)


def _is_todense_callable(val: BackendArg) -> typing.TypeGuard[_TodenseProtocol]:
    """Check whether `val` exposes a `.todense` method.

    Args:
        val (BackendArg): Candidate sparse array.

    Returns:
        bool: True if `val` has a `.todense` attribute.
    """
    return isinstance(val, _TodenseProtocol)


def _is_addable_operand(val: BackendArg) -> typing.TypeGuard[_AddableOperandProtocol]:
    """Check whether `val` exposes `__add__`.

    Args:
        val (BackendArg): Candidate operand.

    Returns:
        bool: True if `val` supports addition.
    """
    return isinstance(val, _AddableOperandProtocol)


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
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg | BackendArg],
    ) -> list[BackendArg]:
        """Convert input structures into sparse or dense array representations.

        Args:
            graph (IRGraph): IR computation graph.
            inputs: Input structures.

        Returns:
            list[object]: Topologically ordered list of input arrays.
        """
        arr_dict: dict[str, BackendArg] = {}
        for k, val in inputs.items():
            raw_val: BackendArg = val
            if isinstance(val, (COOTensor, CSRTensor, CSCTensor)):
                arr_dict[k] = val
            elif _is_sparse_coords(raw_val):
                arr_dict[k] = COOTensor(
                    indices=_SP_BACKEND.asarray(raw_val.coords),
                    values=_SP_BACKEND.asarray(raw_val.data),
                    shape=coerce_shape(raw_val.shape),
                )
            elif _is_shape_holder(raw_val) and not isinstance(val, (list, tuple)):
                dense_val = raw_val.todense() if _is_todense_callable(raw_val) else val
                arr_dict[k] = COOTensor.from_dense(dense_val)
            elif isinstance(val, (list, tuple)):
                np_arr = _SP_BACKEND.asarray(val, dtype=_SP_BACKEND.float32)
                arr_dict[k] = COOTensor.from_dense(np_arr)
            else:
                arr_dict[k] = val

        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in arr_dict for k in input_keys):
            return [arr_dict[k] for k in input_keys]
        return list(arr_dict.values())

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[BackendArg] | None:
        """Compile an IRGraph into an executable sparse function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[TypedVariadicCallable[object]]: Sparse evaluator function.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator

        gen = SparseGenerator(graph)
        code: str = gen.generate()
        ns = load_generated_code(code)
        evaluate_fn = ns.get(gen._func_name)
        return evaluate_fn if callable(evaluate_fn) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg | BackendArg],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile a graph execution on Sparse backend.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs: Named input structures.
            device (Optional[str]): Device target.
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and peak memory metrics.
        """
        del device

        sparse_inputs = self._prepare_inputs(graph, inputs)
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
            """Execute a single evaluation step on sparse backend.

            Returns:
                BackendArg: Evaluated output array.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(sparse_inputs)

            if not sparse_inputs:
                return _SP_BACKEND.zeros((1,), dtype=_SP_BACKEND.float32)

            accum = sparse_inputs[0]
            for t in sparse_inputs[1:]:
                raw_accum: BackendArg = accum
                if _is_addable_operand(raw_accum):
                    accum = raw_accum + t
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
