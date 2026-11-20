"""DPNP backend profiler for runtime execution and memory benchmarking on Intel SYCL devices."""

from __future__ import annotations

import importlib
import logging
import resource
import sys
import time
from collections.abc import Mapping

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort


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


def _get_dpnp_peak_memory_mb() -> float:
    """Retrieve peak memory allocation on Intel SYCL device or host process fallback.

    Returns:
        float: Peak memory in megabytes.
    """
    try:
        dpnp_mod = importlib.import_module("dpnp")
        if adapt_backend_module(dpnp_mod).has_symbol("get_device_memory_info"):
            info: BackendArg = adapt_backend_module(dpnp_mod).require_op("get_device_memory_info")()
            if isinstance(info, (tuple, list)) and len(info) >= 2:
                total, free = info[0], info[1]
                if isinstance(total, (int, float)) and isinstance(free, (int, float)):
                    return float(total - free) / (1024.0 * 1024.0)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return _get_process_memory_mb()


def _safe_wait(waitable: BackendArg) -> None:
    """Invoke .wait() on a SYCL queue or event object while ignoring runtime errors.

    Args:
        waitable (BackendArg): Object potentially exposing a wait() method.
    """
    if waitable is not None and adapt_backend_module(waitable).has_symbol("wait"):
        try:
            adapt_backend_module(waitable).require_op("wait")()
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)


def _sync_dpnp_result(result: BackendArg, sycl_queue: BackendArg | None = None) -> None:
    """Synchronize SYCL execution queue or DPNP array evaluations.

    Args:
        result (BackendArg): Evaluated array or collection of arrays.
        sycl_queue (BackendArg | None): Optional SYCL execution queue.
    """
    _safe_wait(sycl_queue)

    if adapt_backend_module(result).has_symbol("sycl_queue") and adapt_backend_module(
        adapt_backend_module(result).require_symbol("sycl_queue")
    ).has_symbol("wait"):
        _safe_wait(adapt_backend_module(result).require_symbol("sycl_queue"))
    elif adapt_backend_module(result).has_symbol("wait"):
        _safe_wait(result)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_dpnp_result(item, sycl_queue=sycl_queue)
    elif isinstance(result, dict):
        for val in result.values():
            _sync_dpnp_result(val, sycl_queue=sycl_queue)


class DpnpProfiler:
    """Provides execution timing and peak memory profiling for DPNP on Intel SYCL devices."""

    def __init__(self, sycl_queue: BackendArg | None = None) -> None:
        """Initialize DPNP profiler with optional SYCL queue.

        Args:
            sycl_queue (BackendArg | None): Optional target SYCL queue.
        """
        self.sycl_queue: BackendArg | None = sycl_queue

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
    ) -> list[BackendArg]:
        """Convert input payloads into topologically ordered DPNP/NumPy arrays.

        Args:
            graph (IRGraph): IR computation graph.
            inputs: Input mapping.

        Returns:
            list[object]: Ordered list of input arrays.
        """
        dpnp_inputs: dict[str, BackendArg] = {}
        for k, val in inputs.items():
            if type(val).__name__ == "Tensor" and adapt_backend_module(val).has_symbol("data"):
                dpnp_inputs[k] = adapt_backend_module(val).require_symbol("data")
            elif type(val).__name__ == "ndarray" or adapt_backend_module(val).has_symbol("ndim"):
                dpnp_inputs[k] = val
            elif isinstance(val, (list, tuple)):
                from ml_switcheroo_compiler.backends.dpnp.types import asarray

                dpnp_inputs[k] = asarray(val)
            else:
                dpnp_inputs[k] = val

        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in dpnp_inputs for k in input_keys):
            return [dpnp_inputs[k] for k in input_keys]
        return list(dpnp_inputs.values())

    def _compile_runner(self, graph: IRGraph, device: str = "auto") -> TypedVariadicCallable[BackendArg] | None:
        """Compile computational graph into a DPNP executable runner.

        Args:
            graph (IRGraph): Target computation graph.
            device (str): SYCL target device.

        Returns:
            TypedVariadicCallable[object] | None: Executable runner if compiled.
        """
        from ml_switcheroo_compiler.backends.dpnp.generator import DPNPGenerator

        gen = DPNPGenerator(graph, device=device, sycl_queue=self.sycl_queue)
        runner = gen._compile_aot_impl(graph)
        return runner if callable(runner) else None

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | BackendArg],
        device: str = "auto",
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile graph execution on DPNP backend targeting Intel SYCL.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs: Named inputs.
            device (str): SYCL target device ('auto', 'cpu', 'gpu', 'fpga').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and peak memory metrics.
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
