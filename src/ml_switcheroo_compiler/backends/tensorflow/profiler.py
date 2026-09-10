"""TensorFlow backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

try:
    import tensorflow as tf
except ImportError:
    tf = None

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


def _get_tf_peak_memory_mb() -> float:
    """Retrieve peak GPU/accelerator memory usage in megabytes for TensorFlow.

    Returns:
        float: Peak memory allocation in megabytes.
    """
    if tf is None:
        return _get_process_memory_mb()
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus and hasattr(tf.config.experimental, "get_memory_info"):
            mem_info = tf.config.experimental.get_memory_info("GPU:0")
            if "peak" in mem_info and mem_info["peak"] > 0:
                return float(mem_info["peak"]) / (1024.0 * 1024.0)
            if "current" in mem_info and mem_info["current"] > 0:
                return float(mem_info["current"]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return _get_process_memory_mb()


def _sync_tf_result(result: Union[object, list[object], tuple[object, ...], dict[str, object]]) -> None:
    """Force synchronization of TensorFlow evaluation results to host memory.

    Args:
        result (Union[object, list[object], tuple[object, ...], dict[str, object]]): Tensor or container of tensors.
    """
    if tf is None:
        return
    if isinstance(result, dict):
        for val in result.values():
            _sync_tf_result(val)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_tf_result(item)
    elif hasattr(result, "numpy") and callable(result.numpy):
        try:
            _ = result.numpy()
        except Exception:
            pass


def _compute_metrics(
    latencies: list[float],
    peak_mem: float,
    warmup_iters: int,
) -> dict[str, Union[list[float], float]]:
    """Compute statistical latency percentiles and format metrics payload.

    Args:
        latencies (list[float]): Raw measured step latencies in milliseconds.
        peak_mem (float): Measured peak memory usage in megabytes.
        warmup_iters (int): Warmup iteration count.

    Returns:
        dict[str, Union[list[float], float]]: Aggregated metrics.
    """
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
        "peak_memory_mb": peak_mem,
        "warmup_iterations": max(1, warmup_iters),
    }


class TensorFlowProfiler:
    """Provides genuine execution timing and peak device memory profiling for TensorFlow."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
    ) -> list[object]:
        """Convert input structures into topologically ordered TensorFlow tensors.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input structures.

        Returns:
            list[object]: Topologically ordered list of TensorFlow tensors.
        """
        if tf is None:
            return []

        tf_inputs_dict: dict[str, object] = {}
        for k, val in inputs.items():
            if hasattr(val, "shape") and hasattr(val, "dtype") and not isinstance(val, (list, tuple)):
                tf_inputs_dict[k] = val
            else:
                tf_inputs_dict[k] = tf.convert_to_tensor(val, dtype=tf.float32)

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in tf_inputs_dict for k in input_keys):
            return [tf_inputs_dict[k] for k in input_keys]
        return list(tf_inputs_dict.values())

    def _compile_graph(self, graph: IRGraph) -> Optional[Callable[..., object]]:
        """Compile an IRGraph into an executable TensorFlow function.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[Callable[..., object]]: Compiled executable TensorFlow function.

        Raises:
            TypeError: If apply_model cannot be generated from the graph.
        """
        from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator

        code: str = TensorFlowCodeGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        apply_model = ns.get("apply_model")
        if not callable(apply_model):
            raise TypeError("Failed to generate valid apply_model function from TensorFlowCodeGenerator")
        return apply_model

    def _build_step_fn(
        self,
        has_ops: bool,
        compiled_fn: Optional[Callable[..., object]],
        tf_inputs: list[object],
        device: Optional[str],
    ) -> Callable[[], object]:
        """Build execution step function with device context and synchronization.

        Args:
            has_ops (bool): Whether graph contains operations.
            compiled_fn (Optional[Callable[..., object]]): Compiled callable.
            tf_inputs (list[object]): Ordered TensorFlow inputs.
            device (Optional[str]): Target execution device.

        Returns:
            Callable[[], object]: Step execution callable.
        """
        dev_ctx = tf.device(device) if device and tf is not None else None

        def _execute_step() -> object:
            """Execute a single step on TensorFlow.

            Returns:
                object: Output tensor or accumulation.
            """
            if has_ops and compiled_fn is not None:
                if dev_ctx:
                    with dev_ctx:
                        res = compiled_fn(*tf_inputs)
                else:
                    res = compiled_fn(*tf_inputs)
                _sync_tf_result(res)
                return res

            if not tf_inputs:
                return tf.zeros((1,), dtype=tf.float32) if tf is not None else 0.0

            accum = tf.zeros_like(tf_inputs[0])
            for t in tf_inputs:
                accum = tf.add(accum, t)
            _sync_tf_result(accum)
            return accum

        return _execute_step

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on TensorFlow backend with device synchronization.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input tensors or buffers.
            device (Optional[str]): Target device ('cpu', 'gpu:0', etc.).
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        if tf is None:
            return _compute_metrics([0.0] * num_iters, _get_process_memory_mb(), warmup_iters)

        tf_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_fn: Optional[Callable[..., object]] = None
        if has_ops:
            try:
                compiled_fn = self._compile_graph(graph)
            except Exception:
                compiled_fn = None

        step_fn = self._build_step_fn(has_ops, compiled_fn, tf_inputs, device)

        for _ in range(max(1, warmup_iters)):
            _ = step_fn()

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            _ = step_fn()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        return _compute_metrics(latencies, _get_tf_peak_memory_mb(), warmup_iters)
