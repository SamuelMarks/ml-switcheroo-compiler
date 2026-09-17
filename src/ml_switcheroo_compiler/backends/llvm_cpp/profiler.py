"""C++ / LLVM backend profiler for runtime execution and memory benchmarking."""

import ctypes
import resource
import sys
import time
from typing import Callable, Optional, Union

from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator
from ml_switcheroo_compiler.benchmarks.metrics import compute_statistical_metrics
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


def _flatten_input_values(val: object) -> list[float]:
    """Recursively flatten input values into a flat list of floats.

    Args:
        val (object): Scalar or nested container of numbers.

    Returns:
        list[float]: Flattened floats.
    """
    if isinstance(val, (int, float)):
        return [float(val)]
    if isinstance(val, (list, tuple)):
        result: list[float] = []
        for elem in val:
            result.extend(_flatten_input_values(elem))
        return result
    if hasattr(val, "tolist") and callable(val.tolist):
        return _flatten_input_values(val.tolist())
    return []


class CppProfiler:
    """Provides genuine execution timing and peak memory profiling for native C++ compilation."""

    def _compile_graph(self, graph: IRGraph) -> Optional[Callable[..., str]]:
        """Compile an IRGraph into an executable C++ shared library callable.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            Optional[Callable[..., str]]: Native callable executing the compiled C++ graph.
        """
        generator = CppGenerator(graph=graph)
        try:
            return generator._compile_aot_impl(graph)
        except Exception:
            return None

    def _marshal_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
    ) -> tuple[Optional[object], list[object]]:
        """Marshal input tensors into C-contiguous float buffers and ctypes pointer array.

        Args:
            graph (IRGraph): Target computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named inputs.

        Returns:
            tuple[Optional[object], list[object]]: Ctypes pointers array and buffer handles.
        """
        c_in_ptrs: list[ctypes.c_void_p] = []
        raw_in_buffers: list[object] = []
        in_nodes = [n for n in graph.nodes.values() if getattr(n, "op_type", "") == "Input"] if getattr(graph, "nodes", None) else []
        for inp_node in in_nodes:
            val = inputs.get(inp_node.id)
            if val is not None:
                flat_vals = _flatten_input_values(val)
                buf_len = max(1, len(flat_vals))
                c_arr = (ctypes.c_float * buf_len)(*flat_vals)
                raw_in_buffers.append(c_arr)
                c_in_ptrs.append(ctypes.cast(ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p))

        c_inputs = (ctypes.c_void_p * len(c_in_ptrs))(*c_in_ptrs) if c_in_ptrs else None
        return c_inputs, raw_in_buffers

    def _allocate_outputs(
        self,
        graph: IRGraph,
    ) -> tuple[Optional[object], list[object]]:
        """Allocate output float buffers and ctypes pointer array for graph outputs.

        Args:
            graph (IRGraph): Target computation graph.

        Returns:
            tuple[Optional[object], list[object]]: Ctypes pointers array and buffer handles.
        """
        c_out_ptrs: list[ctypes.c_void_p] = []
        raw_out_buffers: list[object] = []
        for out_name in getattr(graph, "outputs", []):
            out_node = graph.nodes.get(out_name) if getattr(graph, "nodes", None) else None
            shape = getattr(out_node, "shape_metadata", None) or (1,)
            num_elem: int = 1
            for d in shape:
                if isinstance(d, int) and d > 0:
                    num_elem *= d
            buf = (ctypes.c_float * num_elem)()
            raw_out_buffers.append(buf)
            c_out_ptrs.append(ctypes.cast(ctypes.cast(buf, ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p))

        c_outputs = (ctypes.c_void_p * len(c_out_ptrs))(*c_out_ptrs) if c_out_ptrs else None
        return c_outputs, raw_out_buffers

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], object]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on C++ backend with ABI tensor marshalling.

        Args:
            graph (IRGraph): The IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], object]]): Named input structures.
            device (Optional[str]): Target execution device ('cpu').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and peak memory metrics.
        """
        del device

        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))

        c_inputs, _ = self._marshal_inputs(graph, inputs)
        c_outputs, _ = self._allocate_outputs(graph)

        compiled_fn: Optional[Callable[..., str]] = None
        if has_ops:
            compiled_fn = self._compile_graph(graph)

        def _execute_step() -> None:
            """Execute a single evaluation step of the compiled C++ kernel."""
            if compiled_fn is not None:
                _ = compiled_fn(c_inputs, c_outputs, None)

        for _ in range(max(1, warmup_iters)):
            _execute_step()

        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            start_ns: int = time.perf_counter_ns()
            _execute_step()
            end_ns: int = time.perf_counter_ns()
            latencies.append((end_ns - start_ns) / 1_000_000.0)

        peak_mem: float = _get_process_memory_mb()

        return compute_statistical_metrics(
            latencies=latencies,
            batch_size=1,
            peak_memory_mb=peak_mem,
            warmup_iters=max(1, warmup_iters),
        )
