"""JAX backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp

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
            if devs and hasattr(devs[0], "memory_stats"):
                stats = devs[0].memory_stats()
                if stats and "peak_bytes_in_use" in stats:
                    peak_mem = float(stats["peak_bytes_in_use"]) / (1024.0 * 1024.0)
                elif stats and "bytes_in_use" in stats:
                    peak_mem = float(stats["bytes_in_use"]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return peak_mem


def _prepare_jax_inputs(
    inputs: dict[str, Union[list[float], list[list[float]], jax.Array]],
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


def _sync_jax_result(result: Union[jax.Array, tuple[object, ...], list[object], dict[str, object], object]) -> None:
    """Force hardware device synchronization on all JAX leaf tensors using block_until_ready.

    Args:
        result (object): Single or multi-output array evaluation result.
    """
    if hasattr(jax, "tree_util") and hasattr(jax.tree_util, "tree_leaves"):
        try:
            leaves = jax.tree_util.tree_leaves(result)
            for leaf in leaves:
                if hasattr(leaf, "block_until_ready") and callable(leaf.block_until_ready):
                    leaf.block_until_ready()
            return
        except Exception:
            pass
    if isinstance(result, dict):
        for val in result.values():
            _sync_jax_result(val)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_jax_result(item)
    elif hasattr(result, "block_until_ready") and callable(result.block_until_ready):
        result.block_until_ready()


class JAXProfiler:
    """Provides genuine execution timing and peak memory profiling for JAX."""

    def _compile_graph(self, graph: IRGraph) -> Callable[..., object]:
        """Compile an IRGraph into an executable JAX callable.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Callable[..., object]: JIT compiled forward callable.

        Raises:
            TypeError: If apply_model cannot be generated from IRGraph.
        """
        from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator

        code: str = JAXCodeGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        native_apply_model = ns.get("apply_model")
        if not callable(native_apply_model):
            raise TypeError("Failed to generate valid apply_model function from JAXCodeGenerator")

        def _raw_forward_step(*args: jax.Array) -> object:
            """Evaluate forward step of graph on JAX backend natively.

            Args:
                *args (jax.Array): Positional tensor inputs.

            Returns:
                object: Evaluated output array or collection.
            """
            return native_apply_model({}, *args)

        try:
            return jax.jit(_raw_forward_step)
        except Exception:
            return _raw_forward_step

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
        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys: list[str] = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in jax_inputs_dict for k in input_keys):
            return [jax_inputs_dict[k] for k in input_keys]
        return list(jax_inputs_dict.values())

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], jax.Array]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
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

        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))
        jax_inputs_dict, _ = _prepare_jax_inputs(inputs)
        jax_input_list: list[jax.Array] = self._order_inputs(graph, jax_inputs_dict)
        compiled_fn: Optional[Callable[..., object]] = self._compile_graph(graph) if has_ops else None

        def _execute_step() -> object:
            """Execute a single forward evaluation step on JAX backend.

            Returns:
                object: Single or collection of evaluated JAX arrays.
            """
            if has_ops and compiled_fn is not None:
                return compiled_fn(*jax_input_list)
            if not jax_input_list:
                return jnp.zeros((1,), dtype=jnp.float32)
            accum = jnp.zeros_like(jax_input_list[0])
            for t in jax_input_list:
                accum = jnp.add(accum, t)
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
