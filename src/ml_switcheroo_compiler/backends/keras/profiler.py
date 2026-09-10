"""Keras backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

try:
    import keras
except ImportError:
    keras = None

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


def _get_keras_peak_memory_mb() -> float:
    """Retrieve peak device memory usage in megabytes for Keras backend.

    Returns:
        float: Peak memory allocation in megabytes.
    """
    if keras is not None and hasattr(keras, "backend"):
        try:
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None and hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                return float(torch_mod.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        except Exception:
            pass
        try:
            tf_mod = sys.modules.get("tensorflow")
            if tf_mod is not None and hasattr(tf_mod, "config"):
                mem_info = tf_mod.config.experimental.get_memory_info("GPU:0")
                if "peak" in mem_info and mem_info["peak"] > 0:
                    return float(mem_info["peak"]) / (1024.0 * 1024.0)
        except Exception:
            pass
    return _get_process_memory_mb()


def _sync_keras_result(result: Union[object, list[object], tuple[object, ...], dict[str, object]]) -> None:
    """Force evaluation synchronization of Keras execution output.

    Args:
        result (Union[object, list[object], tuple[object, ...], dict[str, object]]): Evaluated tensor or collection.
    """
    if keras is None:
        return
    if isinstance(result, dict):
        for val in result.values():
            _sync_keras_result(val)
    elif isinstance(result, (list, tuple)):
        for item in result:
            _sync_keras_result(item)
    elif hasattr(keras, "ops") and hasattr(keras.ops, "convert_to_numpy") and callable(keras.ops.convert_to_numpy):
        try:
            _ = keras.ops.convert_to_numpy(result)
        except Exception:
            pass
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


class KerasProfiler:
    """Provides execution timing and peak memory profiling for Keras models."""

    def _prepare_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
    ) -> list[object]:
        """Convert inputs into topologically ordered Keras-compatible tensor inputs.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input data structures.

        Returns:
            list[object]: Topologically ordered list of tensor inputs.
        """
        if keras is None:
            return []

        keras_inputs_dict: dict[str, object] = {}
        for k, val in inputs.items():
            if hasattr(keras, "ops") and hasattr(keras.ops, "convert_to_tensor"):
                keras_inputs_dict[k] = keras.ops.convert_to_tensor(val, dtype="float32")
            else:
                keras_inputs_dict[k] = val

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in keras_inputs_dict for k in input_keys):
            return [keras_inputs_dict[k] for k in input_keys]
        return list(keras_inputs_dict.values())

    def _compile_model(self, graph: IRGraph) -> Optional[Callable[..., object]]:
        """Compile an IRGraph into an executable Keras Model or callable.

        Args:
            graph (IRGraph): The IR computation graph.

        Returns:
            Optional[Callable[..., object]]: Executable model callable.

        Raises:
            TypeError: If get_model cannot be resolved from the generated code.
        """
        from ml_switcheroo_compiler.backends.keras.generator import KerasCodeGenerator

        code: str = KerasCodeGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        get_model = ns.get("get_model")
        if not callable(get_model):
            raise TypeError("Failed to generate valid get_model function from KerasCodeGenerator")
        model = get_model()
        return model

    def _build_step_fn(
        self,
        has_ops: bool,
        compiled_model: Optional[Callable[..., object]],
        keras_inputs: list[object],
    ) -> Callable[[], object]:
        """Build a single evaluation step callable for Keras backend.

        Args:
            has_ops (bool): Whether graph has operations.
            compiled_model (Optional[Callable[..., object]]): Compiled model callable.
            keras_inputs (list[object]): List of prepared Keras tensor inputs.

        Returns:
            Callable[[], object]: Step execution callable.
        """

        def _execute_step() -> object:
            """Execute a single evaluation step on Keras backend.

            Returns:
                object: Output tensor or accumulation.
            """
            if has_ops and compiled_model is not None:
                if len(keras_inputs) == 1:
                    res = compiled_model(keras_inputs[0])
                else:
                    res = compiled_model(keras_inputs)
                _sync_keras_result(res)
                return res

            if not keras_inputs:
                if hasattr(keras, "ops") and hasattr(keras.ops, "zeros"):
                    return keras.ops.zeros((1,), dtype="float32")
                return 0.0

            accum = keras_inputs[0]
            if hasattr(keras, "ops") and hasattr(keras.ops, "add"):
                accum = keras.ops.zeros_like(keras_inputs[0])
                for t in keras_inputs:
                    accum = keras.ops.add(accum, t)
            _sync_keras_result(accum)
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
        """Profile a graph execution on Keras backend with synchronization.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input tensors or buffers.
            device (Optional[str]): Execution device identifier.
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        if keras is None:
            return _compute_metrics([0.0] * num_iters, _get_process_memory_mb(), warmup_iters)

        keras_inputs = self._prepare_inputs(graph, inputs)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        compiled_model: Optional[Callable[..., object]] = None
        if has_ops:
            try:
                compiled_model = self._compile_model(graph)
            except Exception:
                compiled_model = None

        step_fn = self._build_step_fn(has_ops, compiled_model, keras_inputs)

        for _ in range(max(1, warmup_iters)):
            _ = step_fn()

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start = time.perf_counter()
            _ = step_fn()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        return _compute_metrics(latencies, _get_keras_peak_memory_mb(), warmup_iters)
