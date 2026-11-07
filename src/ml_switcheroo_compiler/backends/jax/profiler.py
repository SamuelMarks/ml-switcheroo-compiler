"""JAX backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import importlib.util
import logging
import resource
import sys
import time
import typing

from ml_switcheroo_compiler.backends.eager_registry import EagerValue, ensure_callable, ensure_eager_tensor
from ml_switcheroo_compiler.backends.protocols import has_backend_member
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort
from ml_switcheroo_compiler.ir.core import IRGraph

if typing.TYPE_CHECKING:
    import jax
    import jax.numpy as jnp

_has_jax = sys.modules.get("jax") is not None or ("jax" not in sys.modules and importlib.util.find_spec("jax") is not None)
if not typing.TYPE_CHECKING:  # pragma: no cover
    jax = importlib.import_module("jax") if _has_jax else None
    jnp = importlib.import_module("jax.numpy") if _has_jax else None


class _MemoryStatsDeviceProtocol(typing.Protocol):
    """Protocol for JAX device objects exposing `memory_stats()`."""

    def memory_stats(self) -> dict[str, int | float] | None:
        """Return device memory statistics dictionary.

        Returns:
            dict[str, int | float] | None: Memory stats mapping.
        """
        ...


class _BlockUntilReadyProtocol(typing.Protocol):
    """Protocol for JAX array objects exposing `block_until_ready()`."""

    def block_until_ready(self) -> EagerValue:
        """Block until asynchronous device computation completes.

        Returns:
            EagerValue: Synchronized array.
        """
        ...


def _is_memory_stats_device(val: EagerValue) -> typing.TypeGuard[_MemoryStatsDeviceProtocol]:
    """Check whether `val` exposes a `.memory_stats` attribute.

    Args:
        val (EagerValue): Candidate JAX device object.

    Returns:
        bool: True if `val` has a `.memory_stats` attribute.
    """
    return has_backend_member(val, "memory_stats")


def _is_block_until_ready(val: EagerValue) -> typing.TypeGuard[_BlockUntilReadyProtocol]:
    """Check whether `val` exposes a `.block_until_ready` attribute.

    Args:
        val (EagerValue): Candidate JAX array object.

    Returns:
        bool: True if `val` has a `.block_until_ready` attribute.
    """
    return has_backend_member(val, "block_until_ready")


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
        default_backend = jax.default_backend()
        if default_backend in ("gpu", "tpu", "cuda"):
            devs = jax.devices()
            if devs and _is_memory_stats_device(devs[0]):
                stats = devs[0].memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    peak_mem = float(stats["peak_bytes_in_use"]) / (1024.0 * 1024.0)
                elif stats and "bytes_in_use" in stats:
                    peak_mem = float(stats["bytes_in_use"]) / (1024.0 * 1024.0)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
        logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
    return peak_mem


def _prepare_jax_inputs(
    inputs: dict[str, list[float] | list[list[float]] | jax.Array],
) -> tuple[dict[str, jax.Array], list[jax.Array]]:
    """Convert input values to JAX array representations.

    Args:
        inputs (dict[str, Union[list[float], list[list[float]], jax.Array]]): Named input structures.

    Returns:
        tuple[dict[str, jax.Array], list[jax.Array]]: Mapping and flattened list of JAX arrays.
    """
    jax_inputs_dict: dict[str, jax.Array] = {}
    for k, val in inputs.items():
        if isinstance(val, jax.Array):
            jax_inputs_dict[k] = val
        else:
            jax_inputs_dict[k] = jnp.asarray(val, dtype=jnp.float32)
    return jax_inputs_dict, list(jax_inputs_dict.values())


def _sync_jax_result(result: jax.Array | tuple[EagerValue, ...] | list[EagerValue] | dict[str, EagerValue] | EagerValue) -> None:
    """Force hardware device synchronization on all JAX leaf tensors using block_until_ready.

    Args:
        result (Union[jax.Array, tuple[EagerValue, ...], list[EagerValue], dict[str, EagerValue], EagerValue]): Single or multi-output array evaluation result.
    """
    if has_backend_member(jax, "tree_util") and has_backend_member(jax.tree_util, "tree_leaves"):
        try:
            leaves = jax.tree_util.tree_leaves(result)
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

    def _compile_graph(self, graph: IRGraph) -> TypedVariadicCallable[EagerValue]:
        """Compile an IRGraph into an executable JAX callable.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            TypedVariadicCallable[object]: JIT compiled forward callable.

        Raises:
            TypeError: If apply_model cannot be generated from IRGraph.
        """
        from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator

        code: str = JAXCodeGenerator(graph).generate()
        ns: dict[str, EagerValue] = {}
        exec(code, ns)
        native_apply_model = ns.get("apply_model")
        if not callable(native_apply_model):
            raise TypeError("Failed to generate valid apply_model function from JAXCodeGenerator")

        def _raw_forward_step(*args: jax.Array) -> EagerValue:
            """Evaluate forward step of graph on JAX backend natively.

            Args:
                *args (jax.Array): Positional tensor inputs.

            Returns:
                EagerValue: Evaluated output array or collection.
            """
            return native_apply_model({}, *args)

        try:
            return ensure_callable(jax.jit(_raw_forward_step))
        except Exception:
            return ensure_callable(_raw_forward_step)

    def _order_inputs(
        self,
        graph: IRGraph,
        jax_inputs_dict: dict[str, jax.Array],
    ) -> list[jax.Array]:
        """Order inputs topologically to match generated model parameter layout.

        Args:
            graph (IRGraph): IR computation graph.
            jax_inputs_dict (dict[str, jax.Array]): Dict of JAX input arrays.

        Returns:
            list[jax.Array]: Topologically ordered list of JAX arrays.
        """
        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys: list[str] = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in jax_inputs_dict for k in input_keys):
            return [jax_inputs_dict[k] for k in input_keys]
        return list(jax_inputs_dict.values())

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, list[float] | list[list[float]] | jax.Array],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, list[float] | float]:
        """Profile a graph execution on JAX with device synchronization.

        Args:
            graph (IRGraph): The IR computation graph to execute.
            inputs (dict[str, Union[list[float], list[list[float]], jax.Array]]): Named input arrays.
            device (Optional[str]): Device target ('cpu', 'gpu', 'tpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and memory metrics.
        """
        del device

        has_ops: bool = bool(graph.nodes and any(n.op_type != "Input" for n in graph.nodes.values()))
        jax_inputs_dict, _ = _prepare_jax_inputs(inputs)
        jax_input_list: list[jax.Array] = self._order_inputs(graph, jax_inputs_dict)
        compiled_fn: TypedVariadicCallable[EagerValue] | None = self._compile_graph(graph) if has_ops else None

        def _execute_step() -> EagerValue:
            """Execute a single forward evaluation step on JAX backend.

            Returns:
                EagerValue: Single or collection of evaluated JAX arrays.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(*jax_input_list)
            if not jax_input_list:
                return ensure_eager_tensor(jnp.zeros((1,), dtype=jnp.float32))
            accum = jnp.zeros_like(jax_input_list[0])
            for t in jax_input_list:
                accum = jnp.add(accum, t)
            return ensure_eager_tensor(accum)

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
