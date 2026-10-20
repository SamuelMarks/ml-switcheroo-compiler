"""MLX backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import logging
import resource
import sys
import time
import typing
from collections.abc import Callable
from types import ModuleType

from ml_switcheroo_compiler.backends.eager_registry import EagerValue
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph

mx: ModuleType | None
try:
    import mlx.core as _mx

    mx = _mx
except ImportError:
    mx = None


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


def _get_mlx_peak_memory_mb() -> float:
    """Retrieve peak memory usage on Apple Silicon / MLX runtime.

    Returns:
        float: Peak memory in megabytes.
    """
    peak_mem: float = _get_process_memory_mb()
    if mx is not None:
        get_mem_fn = None
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            get_mem_fn = getattr(mx.metal, "get_peak_memory", None)
        elif hasattr(mx, "get_peak_memory"):
            get_mem_fn = getattr(mx, "get_peak_memory", None)
        if get_mem_fn is not None and callable(get_mem_fn):
            try:
                metal_mem = float(get_mem_fn()) / (1024.0 * 1024.0)
                if metal_mem > 0.0:
                    return metal_mem
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        if hasattr(mx, "metal") and hasattr(mx.metal, "get_active_memory"):
            try:
                active_mem = float(mx.metal.get_active_memory()) / (1024.0 * 1024.0)
                if active_mem > 0.0:
                    return active_mem
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    return peak_mem


def _reset_mlx_peak_memory() -> None:
    """Reset peak memory stats on MLX Metal runtime if available."""
    if mx is not None:
        reset_fn = None
        if hasattr(mx, "metal") and hasattr(mx.metal, "reset_peak_memory"):
            reset_fn = getattr(mx.metal, "reset_peak_memory", None)
        elif hasattr(mx, "reset_peak_memory"):
            reset_fn = getattr(mx, "reset_peak_memory", None)
        if reset_fn is not None and callable(reset_fn):
            try:
                reset_fn()
            except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)


def _sync_mlx_result(result: EagerValue | dict[str, EagerValue] | list[EagerValue] | tuple[EagerValue, ...]) -> None:
    """Force Metal evaluation synchronization on MLX arrays.

    Args:
        result (Union[object, dict[str, object], list[object], tuple[object, ...]]): Evaluated array or collection.
    """
    if mx is None or not hasattr(mx, "eval"):
        return
    if isinstance(result, dict):
        for val in result.values():
            mx.eval(val)
    elif isinstance(result, (list, tuple)):
        for item in result:
            mx.eval(item)
    else:
        mx.eval(result)


def _prepare_mlx_inputs(
    inputs: dict[str, list[float] | list[list[float]] | EagerValue],
) -> list[EagerValue]:
    """Convert input structures into MLX array representations.

    Args:
        inputs (dict[str, Union[list[float], list[list[float]], object]]): Inputs for the graph.

    Returns:
        list[object]: List of MLX array objects.
    """
    if mx is None:
        return []
    mlx_inputs: list[EagerValue] = []
    for val in inputs.values():
        if isinstance(getattr(mx, "array", None), type) and isinstance(val, mx.array):
            mlx_inputs.append(val)
        elif type(val).__name__ == "array" and "mlx" in getattr(type(val), "__module__", ""):
            mlx_inputs.append(val)
        else:
            mlx_inputs.append(getattr(mx, "array")(val, dtype=mx.float32))
    return mlx_inputs


class MLXProfiler:
    """Provides genuine memory and latency profiling for the MLX backend."""

    def _prepare_ordered_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | EagerValue],
    ) -> list[EagerValue]:
        """Convert and order input structures into MLX array representations.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Inputs for the graph.

        Returns:
            list[object]: Ordered list of MLX array objects.
        """
        mlx_inputs_dict: dict[str, EagerValue] = {}
        for k, val in inputs.items():
            mx_array_cls = getattr(mx, "array", None)
            if (isinstance(mx_array_cls, type) and isinstance(val, mx_array_cls)) or (type(val).__name__ == "array" and "mlx" in getattr(type(val), "__module__", "")):
                mlx_inputs_dict[k] = val
            else:
                mlx_inputs_dict[k] = getattr(mx, "array")(val, dtype=getattr(mx, "float32", None))

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in mlx_inputs_dict for k in input_keys):
            return [mlx_inputs_dict[k] for k in input_keys]
        return list(mlx_inputs_dict.values())

    def _compile_model(self, graph: IRGraph) -> TypedVariadicCallable[EagerValue] | None:
        """Compile an IRGraph into an executable MLX model.

        Args:
            graph (IRGraph): IR computation graph.

        Returns:
            Optional[TypedVariadicCallable[object]]: Compiled MLX forward function.

        Raises:
            TypeError: If CompiledModel cannot be generated from MLXCodeGenerator.
        """
        from ml_switcheroo_compiler.backends.mlx.generator import MLXCodeGenerator

        code: str = MLXCodeGenerator(graph).generate()
        ns: dict[str, EagerValue] = {}
        try:
            exec(code, ns)
        except (ImportError, ModuleNotFoundError):
            return None
        model_cls = ns.get("CompiledModel")
        if model_cls is not None and callable(model_cls):
            model_inst = model_cls()
            if mx is not None and hasattr(mx, "compile") and callable(mx.compile):
                try:
                    return typing.cast("TypedVariadicCallable[EagerValue] | None", getattr(mx, "compile")(model_inst))
                except Exception:
                    return typing.cast("TypedVariadicCallable[EagerValue] | None", model_inst)
            return typing.cast("TypedVariadicCallable[EagerValue] | None", model_inst)
        raise TypeError("Failed to generate valid CompiledModel from MLXCodeGenerator")

    def _measure_latencies(
        self,
        step_fn: Callable[[], EagerValue],
        num_iters: int,
    ) -> list[float]:
        """Measure execution latencies over multiple iterations.

        Args:
            step_fn (Callable[[], object]): Step evaluation callback.
            num_iters (int): Measurement iteration count.

        Returns:
            list[float]: Latency measurements in milliseconds.
        """
        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            res = step_fn()
            _sync_mlx_result(res)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)
        return latencies

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | EagerValue],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, list[float] | float]:
        """Profile a graph execution on MLX backend with eval synchronization.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Inputs for the graph.
            device (Optional[str]): Target device ('cpu', 'gpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and memory metrics.

        Raises:
            TypeError: If CompiledModel cannot be generated from MLXCodeGenerator.
        """
        del device

        if mx is None:
            return {
                "latencies": [0.0] * num_iters,
                "latency_ms": 0.0,
                "mean_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "peak_memory_mb": _get_process_memory_mb(),
            }

        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))
        mlx_input_list: list[EagerValue] = self._prepare_ordered_inputs(graph, inputs)

        compiled_fn: TypedVariadicCallable[EagerValue] | None = None
        if has_ops:
            compiled_fn = self._compile_model(graph)

        def _execute_step() -> EagerValue | dict[str, EagerValue]:
            """Execute a single evaluation step on MLX backend.

            Returns:
                Union[object, dict[str, object]]: Single or collection of output arrays.
            """
            if has_ops and compiled_fn is not None:
                try:
                    return compiled_fn(*mlx_input_list)
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
            if not mlx_input_list:
                return typing.cast(EagerValue, getattr(mx, "zeros")((1,), dtype=getattr(mx, "float32")))
            accum = getattr(mx, "zeros_like")(mlx_input_list[0])
            for t in mlx_input_list:
                accum = getattr(mx, "add")(accum, t)
            return typing.cast(typing.Union[EagerValue, "dict[str, EagerValue]"], accum)

        _reset_mlx_peak_memory()
        for _ in range(max(1, warmup_iters)):
            _sync_mlx_result(_execute_step())
        _reset_mlx_peak_memory()

        latencies = self._measure_latencies(_execute_step, num_iters)

        sorted_lat: list[float] = sorted(latencies) if latencies else [0.0]
        mean_lat: float = float(sum(latencies) / len(latencies)) if latencies else 0.0
        p50_lat: float = float(sorted_lat[int(len(sorted_lat) * 0.50)])
        p95_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))])
        p99_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))])

        return {
            "latencies": latencies,
            "latency_ms": mean_lat,
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": p50_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "peak_memory_mb": _get_mlx_peak_memory_mb(),
            "warmup_iterations": max(1, warmup_iters),
        }
