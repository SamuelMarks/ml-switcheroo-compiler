"""PyTorch backend profiler for runtime execution and memory benchmarking."""

import resource
import sys
import time
from typing import Callable, Optional, Union

import torch

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


class TorchGraphModule(torch.nn.Module):
    """Executable torch.nn.Module compiling and executing an IRGraph natively."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize TorchGraphModule with the IR computation graph.

        Args:
            graph (IRGraph): The computation graph to encapsulate.

        Raises:
            TypeError: If the generated model class is invalid or missing.
        """
        super().__init__()
        self.graph: IRGraph = graph
        from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator

        code: str = PyTorchCodeGenerator(graph).generate()
        ns: dict[str, object] = {}
        exec(code, ns)
        model_cls = ns.get("CompiledModel")
        if model_cls is not None and isinstance(model_cls, type) and issubclass(model_cls, torch.nn.Module):
            self.model: torch.nn.Module = model_cls()
        else:
            raise TypeError("Failed to generate valid torch.nn.Module from IRGraph")

    def forward(self, *args: Union[torch.Tensor, dict[str, torch.Tensor]], **kwargs: torch.Tensor) -> Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        """Execute the native compiled graph forward pass.

        Args:
            *args (Union[torch.Tensor, dict[str, torch.Tensor]]): Positional inputs or input dict.
            **kwargs (torch.Tensor): Named tensor inputs.

        Returns:
            Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]: Evaluated output tensors.
        """
        if len(args) == 1 and isinstance(args[0], dict):
            inp_dict: dict[str, torch.Tensor] = args[0]
            sorted_nodes = topological_sort(self.graph)
            input_keys: list[str] = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
            if input_keys and all(k in inp_dict for k in input_keys):
                ordered = [inp_dict[k] for k in input_keys]
            else:
                ordered = list(inp_dict.values())
            res = self.model(*ordered)
            if isinstance(res, torch.Tensor) and self.graph.outputs:
                return {self.graph.outputs[0]: res}
            return res
        return self.model(*args, **kwargs)


class PyTorchProfiler:
    """Provides genuine execution timing and peak device memory profiling for PyTorch."""

    def _resolve_device(self, device: Optional[str]) -> torch.device:
        """Resolve valid PyTorch execution device with hardware fallbacks.

        Args:
            device (Optional[str]): Desired device string.

        Returns:
            torch.device: Validated torch device instance.
        """
        target = device or "cpu"
        if target == "cuda" and not torch.cuda.is_available():
            target = "cpu"
        elif target == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            target = "cpu"
        return torch.device(target)

    def _prepare_inputs(
        self,
        inputs: dict[str, Union[list[float], list[list[float]], torch.Tensor]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Convert input payloads to device tensors.

        Args:
            inputs (dict[str, Union[list[float], list[list[float]], torch.Tensor]]): Input dict.
            device (torch.device): Target device.

        Returns:
            list[torch.Tensor]: Converted tensor list.
        """
        torch_inputs: list[torch.Tensor] = []
        for val in inputs.values():
            if isinstance(val, torch.Tensor):
                torch_inputs.append(val.to(device))
            else:
                torch_inputs.append(torch.tensor(val, device=device, dtype=torch.float32))
        return torch_inputs

    def _prepare_ordered_inputs(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], torch.Tensor]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Prepare inputs ordered topologically for compiled forward execution.

        Args:
            graph (IRGraph): IR computation graph.
            inputs (dict[str, Union[list[float], list[list[float]], torch.Tensor]]): Input mapping.
            device (torch.device): Target PyTorch device.

        Returns:
            list[torch.Tensor]: Ordered tensor inputs list.
        """
        torch_inputs_dict: dict[str, torch.Tensor] = {}
        for k, val in inputs.items():
            if isinstance(val, torch.Tensor):
                torch_inputs_dict[k] = val.to(device)
            else:
                torch_inputs_dict[k] = torch.tensor(val, device=device, dtype=torch.float32)

        sorted_nodes = topological_sort(graph) if getattr(graph, "nodes", None) else []
        input_keys = [n.id for n in sorted_nodes if getattr(n, "op_type", "") == "Input"]
        if input_keys and all(k in torch_inputs_dict for k in input_keys):
            return [torch_inputs_dict[k] for k in input_keys]
        return list(torch_inputs_dict.values())

    def _sync(self, device: torch.device) -> None:
        """Synchronize execution across hardware devices.

        Args:
            device (torch.device): Device to synchronize.
        """
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.synchronize()

    def _reset_cuda_stats(self, device: torch.device) -> None:
        """Reset peak memory stats on CUDA device if available.

        Args:
            device (torch.device): Target PyTorch device.
        """
        if device.type == "cuda" and torch.cuda.is_available() and hasattr(torch.cuda, "reset_peak_memory_stats"):
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

    def _compile_module(
        self,
        graph: IRGraph,
        device: torch.device,
        input_list: list[torch.Tensor],
    ) -> Optional[torch.nn.Module]:
        """Compile an IRGraph into an optimized PyTorch module.

        Args:
            graph (IRGraph): The IR graph to compile.
            device (torch.device): Target execution device.
            input_list (list[torch.Tensor]): Input tensor examples for tracing.

        Returns:
            Optional[torch.nn.Module]: Compiled PyTorch module.
        """
        base_module = TorchGraphModule(graph).to(device)
        compiled: torch.nn.Module = base_module
        if callable(getattr(torch, "compile", None)):
            try:
                compiled = torch.compile(base_module)
            except Exception:
                compiled = base_module
        if compiled is base_module and hasattr(torch, "jit") and callable(getattr(torch.jit, "trace", None)) and input_list:
            try:
                compiled = torch.jit.trace(base_module, tuple(input_list))
            except Exception:
                compiled = base_module
        return compiled

    def _measure_latencies(
        self,
        device: torch.device,
        step_fn: Callable[[], object],
        num_iters: int,
    ) -> list[float]:
        """Measure execution latencies over multiple iterations with device sync.

        Args:
            device (torch.device): Execution device.
            step_fn (Callable[[], object]): Forward evaluation callback.
            num_iters (int): Measurement iteration count.

        Returns:
            list[float]: Latency measurements in milliseconds.
        """
        is_cuda: bool = bool(device.type == "cuda" and torch.cuda.is_available())
        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            cuda_timed: bool = False
            if is_cuda and hasattr(torch.cuda, "Event"):
                try:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    _ = step_fn()
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(float(start_event.elapsed_time(end_event)))
                    cuda_timed = True
                except Exception:
                    cuda_timed = False
            if not cuda_timed:
                self._sync(device)
                start = time.perf_counter()
                _ = step_fn()
                self._sync(device)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)
        return latencies

    def _extract_peak_memory_mb(self, device: torch.device) -> float:
        """Extract peak device memory in megabytes with hardware-specific APIs.

        Args:
            device (torch.device): PyTorch execution device.

        Returns:
            float: Peak memory in megabytes.
        """
        if device.type == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))
        if device.type == "mps":
            if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
                try:
                    mps_mem = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
                    if mps_mem > 0.0:
                        return mps_mem
                except Exception:
                    pass
        return _get_process_memory_mb()

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, Union[list[float], list[list[float]], torch.Tensor]],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, Union[list[float], float]]:
        """Profile a graph execution on the PyTorch backend with hardware synchronization.

        Args:
            graph (IRGraph): The IR computation graph to execute.
            inputs (dict[str, Union[list[float], list[list[float]], torch.Tensor]]): Named input tensors or buffers.
            device (Optional[str]): Device target ('cpu', 'cuda', 'mps').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            dict[str, Union[list[float], float]]: Latency and memory metrics.
        """
        torch_dev = self._resolve_device(device)
        has_ops: bool = bool(getattr(graph, "nodes", None) and any(getattr(n, "op_type", "") != "Input" for n in graph.nodes.values()))
        torch_input_list: list[torch.Tensor] = self._prepare_ordered_inputs(graph, inputs, torch_dev)

        compiled_module: Optional[torch.nn.Module] = None
        if has_ops:
            compiled_module = self._compile_module(graph, torch_dev, torch_input_list)

        def _execute_step() -> Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
            """Execute a single evaluation step on PyTorch backend.

            Returns:
                Union[torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]: Single or collection of output tensors.
            """
            if has_ops and compiled_module is not None:
                return compiled_module(*torch_input_list)
            if not torch_input_list:
                return torch.zeros((1,), device=torch_dev, dtype=torch.float32)
            accum = torch.zeros_like(torch_input_list[0])
            for t in torch_input_list:
                accum = torch.add(accum, t)
            return accum

        self._reset_cuda_stats(torch_dev)
        for _ in range(max(1, warmup_iters)):
            _ = _execute_step()
            self._sync(torch_dev)
        self._reset_cuda_stats(torch_dev)

        latencies = self._measure_latencies(torch_dev, _execute_step, num_iters)

        sorted_lat: list[float] = sorted(latencies) if latencies else [0.0]
        mean_lat: float = float(sum(latencies) / len(latencies)) if latencies else 0.0
        p50_lat: float = float(sorted_lat[int(len(sorted_lat) * 0.50)])
        p95_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))])
        p99_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))])
        peak_mem: float = self._extract_peak_memory_mb(torch_dev)

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
