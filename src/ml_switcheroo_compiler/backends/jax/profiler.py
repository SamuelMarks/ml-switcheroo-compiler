"""JAX backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import resource
import sys
import time
import types
import typing
from collections.abc import Mapping

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_operand_list,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

if typing.TYPE_CHECKING:
    import jax as _jax_types

_has_jax = sys.modules.get("jax") is not None or (
    "jax" not in sys.modules and importlib.util.find_spec("jax") is not None
)
jax: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("jax") if _has_jax else None
jnp: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("jax.numpy") if _has_jax else None


@typing.runtime_checkable
class _MemoryStatsDeviceProtocol(typing.Protocol):
    """Protocol for JAX device objects exposing `memory_stats()`."""

    def memory_stats(self) -> dict[str, int | float] | None:
        """Return device memory statistics dictionary.

        Returns:
            ProfileMetricsDict: Memory stats mapping.
        """
        ...


@typing.runtime_checkable
class _BlockUntilReadyProtocol(typing.Protocol):
    """Protocol for JAX array objects exposing `block_until_ready()`."""

    def block_until_ready(self) -> BackendArg:
        """Block until asynchronous device computation completes.

        Returns:
            BackendArg: Synchronized array.
        """
        ...


def _is_memory_stats_device(val: BackendArg) -> typing.TypeGuard[_MemoryStatsDeviceProtocol]:
    """Check whether `val` exposes a `.memory_stats` attribute.

    Args:
        val (BackendArg): Candidate JAX device object.

    Returns:
        bool: True if `val` conforms to `_MemoryStatsDeviceProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _MemoryStatsDeviceProtocol)
    return found or is_mock_with_attrs(val, "memory_stats")


def _is_block_until_ready(val: BackendArg) -> typing.TypeGuard[_BlockUntilReadyProtocol]:
    """Check whether `val` exposes a `.block_until_ready` attribute.

    Args:
        val (BackendArg): Candidate JAX array object.

    Returns:
        bool: True if `val` conforms to `_BlockUntilReadyProtocol`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, _BlockUntilReadyProtocol)
    return found or is_mock_with_attrs(val, "block_until_ready")


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


def _extract_jax_peak_memory_mb() -> float:
    """Extract peak device memory for JAX if running on GPU/TPU accelerator.

    Returns:
        float: Peak memory in megabytes.
    """
    peak_mem: float = _get_process_memory_mb()
    try:
        default_backend = adapt_backend_module(jax).require_op("default_backend")()
        if default_backend in ("gpu", "tpu", "cuda"):
            devs: BackendArg = adapt_backend_module(jax).require_op("devices")()
            if isinstance(devs, (list, tuple)) and devs and _is_memory_stats_device(devs[0]):
                stats = devs[0].memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    peak_mem = float(stats["peak_bytes_in_use"]) / (1024.0 * 1024.0)
                elif stats and "bytes_in_use" in stats:
                    peak_mem = float(stats["bytes_in_use"]) / (1024.0 * 1024.0)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return peak_mem


def _prepare_jax_inputs(
    inputs: Mapping[str, list[float] | list[list[float]] | _jax_types.Array],
) -> tuple[dict[str, BackendArg], list[BackendArg]]:
    """Convert input values to JAX array representations.

    Args:
        inputs: Named input structures.

    Returns:
        tuple[dict[str, BackendArg], list[BackendArg]]: Mapping and flattened list of JAX arrays.
    """
    arr_cls: BackendArg = adapt_backend_module(jax).get_symbol("Array", None)
    jnp_adp = adapt_backend_module(jnp)
    jax_inputs_dict: dict[str, BackendArg] = {}
    for k, val in inputs.items():
        raw_val: BackendArg = val
        if isinstance(arr_cls, type) and isinstance(val, arr_cls):
            jax_inputs_dict[k] = raw_val
        else:
            jax_inputs_dict[k] = jnp_adp.require_op("asarray")(raw_val, dtype=jnp_adp.get_symbol("float32", None))
    return jax_inputs_dict, list(jax_inputs_dict.values())


def _sync_jax_result(
    result: _jax_types.Array | tuple[BackendArg, ...] | list[BackendArg] | dict[str, BackendArg] | BackendArg,
) -> None:
    """Force hardware device synchronization on all JAX leaf tensors using block_until_ready.

    Args:
        result (Union[jax.Array, tuple[BackendArg, ...], list[BackendArg], dict[str, BackendArg], BackendArg]): Single
            or multi-output array evaluation result.
    """
    tree_mod = adapt_backend_module(jax).get_symbol("tree_util", None)
    if tree_mod is not None and adapt_backend_module(tree_mod).has_symbol("tree_leaves"):
        try:
            leaves = coerce_operand_list(adapt_backend_module(tree_mod).require_op("tree_leaves")(result))
            for leaf in leaves:
                if _is_block_until_ready(leaf) and callable(leaf.block_until_ready):
                    leaf.block_until_ready()
            return
        except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
            logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    if isinstance(result, dict):
        for val in result.values():
            _sync_jax_result(val)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_jax_result(item)
    elif _is_block_until_ready(result) and callable(result.block_until_ready):
        result.block_until_ready()


class JAXProfiler:
    """Provides genuine execution timing and peak memory profiling for JAX."""

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[BackendArg]:
        """Compile an IRGraph into an executable JAX callable.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            TypedVariadicCallable[object]: JIT compiled forward callable.

        Raises:
            TypeError: If apply_model cannot be generated from IRGraph.
        """
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator

        code: str = JAXCodeGenerator(graph).generate()
        ns = load_generated_code(code)
        native_apply_model = ns.get("apply_model")
        if not callable(native_apply_model):
            raise TypeError("Failed to generate valid apply_model function from JAXCodeGenerator")
        apply_fn = require_callable(native_apply_model)

        def _raw_forward_step(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Evaluate forward step of graph on JAX backend natively.

            Args:
                *args (BackendArg): Positional tensor inputs.
                **kwargs (BackendArg): Keyword arguments.

            Returns:
                BackendArg: Evaluated output array or collection.
            """
            del kwargs
            return apply_fn({}, *args)

        try:
            jitted = adapt_backend_module(jax).require_op("jit")(_raw_forward_step)
            bound_jit = require_callable(jitted)

            def _jit_step(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Invoke JIT-compiled forward step.

                Args:
                    *args (BackendArg): Positional tensor inputs.
                    **kwargs (BackendArg): Keyword arguments.

                Returns:
                    BackendArg: Evaluated output array or collection.
                """
                del kwargs
                return bound_jit(*args)

            return _jit_step
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
            return _raw_forward_step

    def _order_inputs(
        self,
        graph: IRGraph,
        jax_inputs_dict: dict[str, BackendArg],
    ) -> list[BackendArg]:
        """Order inputs topologically to match generated model parameter layout.

        Args:
            graph (IRGraph): IR computation graph.
            jax_inputs_dict (dict[str, BackendArg]): Dict of JAX input arrays.

        Returns:
            list[BackendArg]: Topologically ordered list of JAX arrays.
        """
        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys: list[str] = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in jax_inputs_dict for k in input_keys):
            return [jax_inputs_dict[k] for k in input_keys]
        return list(jax_inputs_dict.values())

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | _jax_types.Array],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile a graph execution on JAX with device synchronization.

        Args:
            graph (IRGraph): The IR computation graph to execute.
            inputs: Named input arrays.
            device (Optional[str]): Device target ('cpu', 'gpu', 'tpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and memory metrics.
        """
        del device

        has_ops: bool = bool(graph.nodes and any(n.op_type != "Input" for n in graph.nodes.values()))
        jax_inputs_dict, _ = _prepare_jax_inputs(inputs)
        jax_input_list: list[BackendArg] = self._order_inputs(graph, jax_inputs_dict)
        compiled_fn: TypedVariadicCallable[BackendArg] | None = self._compile_graph(graph) if has_ops else None

        def _execute_step() -> BackendArg:
            """Execute a single forward evaluation step on JAX backend.

            Returns:
                BackendArg: Single or collection of evaluated JAX arrays.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(*jax_input_list)
            jnp_step = adapt_backend_module(jnp)
            if not jax_input_list:
                return jnp_step.require_op("zeros")((1,), dtype=jnp_step.get_symbol("float32", None))
            accum: BackendArg = jnp_step.require_op("zeros_like")(jax_input_list[0])
            for t in jax_input_list:
                accum = jnp_step.require_op("add")(accum, t)
            return accum

        for _ in range(max(1, warmup_iters)):
            _sync_jax_result(_execute_step())

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            res = _execute_step()
            _sync_jax_result(res)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        sorted_lat: list[float] = sorted(latencies) if latencies else [0.0]
        mean_lat: float = float(sum(latencies) / len(latencies)) if latencies else 0.0
        p50_lat: float = float(sorted_lat[int(len(sorted_lat) * 0.50)])
        p95_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))])
        p99_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))])
        peak_mem: float = _extract_jax_peak_memory_mb()

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
