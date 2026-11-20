"""PyTorch backend profiler for runtime execution and memory benchmarking."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import resource
import sys
import time
import types
import typing
from collections.abc import Callable, Mapping

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    EagerTensorProtocol,
    coerce_float,
    require_callable,
)
from ml_switcheroo_compiler.backends.protocols import BackendModuleProtocol, adapt_backend_module
from ml_switcheroo_compiler.benchmarks.metrics import ProfileMetricsDict
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from ml_switcheroo_compiler.core.utils.graph_utils import topological_sort

if typing.TYPE_CHECKING:
    import torch as _torch_types

_has_torch = sys.modules.get("torch") is not None or (
    "torch" not in sys.modules and importlib.util.find_spec("torch") is not None
)
torch: types.ModuleType | BackendModuleProtocol | None = importlib.import_module("torch") if _has_torch else None


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


class TorchGraphModule:
    """Executable wrapper around a generated torch.nn.Module compiling and executing an IRGraph natively."""

    def __init__(self, graph: IRGraph) -> None:
        """Initialize TorchGraphModule with the IR computation graph.

        Args:
            graph (IRGraph): The computation graph to encapsulate.

        Raises:
            TypeError: If the generated model class is invalid or missing.
        """
        self.graph: IRGraph = graph
        from ml_switcheroo_compiler.backends._module_loader import load_generated_code
        from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator

        code: str = PyTorchCodeGenerator(graph).generate()
        ns = load_generated_code(code)
        model_cls = ns.get("CompiledModel")
        torch_adp = adapt_backend_module(torch)
        nn_mod = torch_adp.get_symbol("nn", None)
        nn_module_cls: BackendArg = adapt_backend_module(nn_mod).get_symbol("Module", None)
        if (
            model_cls is not None
            and isinstance(model_cls, type)
            and isinstance(nn_module_cls, type)
            and issubclass(model_cls, nn_module_cls)
        ):
            self.model: EagerTensorProtocol = require_callable(model_cls)()
        else:
            raise TypeError("Failed to generate valid torch.nn.Module from IRGraph")

    def to(self, device: _torch_types.device | BackendArg) -> TorchGraphModule:
        """Move the wrapped model to the specified device.

        Args:
            device (Union[torch.device, BackendArg]): Target execution device.

        Returns:
            TorchGraphModule: Self reference after device placement.
        """
        self.model = adapt_backend_module(self.model).require_op("to")(device)
        return self

    def __call__(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Invoke forward pass on the wrapped model.

        Args:
            *args (BackendArg): Positional inputs or input dict.
            **kwargs (BackendArg): Named tensor inputs.

        Returns:
            BackendArg: Evaluated output tensors.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args: BackendArg, **kwargs: BackendArg) -> BackendArg:
        """Execute the native compiled graph forward pass.

        Args:
            *args (BackendArg): Positional inputs or input dict.
            **kwargs (BackendArg): Named tensor inputs.

        Returns:
            BackendArg: Evaluated output tensors.
        """
        tensor_cls: BackendArg = adapt_backend_module(torch).get_symbol("Tensor", None)
        if len(args) == 1 and isinstance(args[0], dict):
            inp_dict = args[0]
            sorted_nodes = topological_sort(self.graph)
            input_keys: list[str] = [n.id for n in sorted_nodes if n.op_type == "Input"]
            if input_keys and all(k in inp_dict for k in input_keys):
                ordered = [inp_dict[k] for k in input_keys]
            else:
                ordered = list(inp_dict.values())
            res: BackendArg = require_callable(self.model)(*ordered)
            raw_res: BackendArg = res
            if isinstance(tensor_cls, type) and isinstance(res, tensor_cls) and self.graph.outputs:
                out_map: dict[str, BackendArg] = {self.graph.outputs[0]: raw_res}
                return out_map
            return raw_res
        return require_callable(self.model)(*args, **kwargs)


class PyTorchProfiler:
    """Provides genuine execution timing and peak device memory profiling for PyTorch."""

    def _resolve_device(self, device: str | None) -> EagerTensorProtocol:
        """Resolve valid PyTorch execution device with hardware fallbacks.

        Args:
            device (Optional[str]): Desired device string.

        Returns:
            EagerTensorProtocol: Validated torch device instance.
        """
        torch_adp = adapt_backend_module(torch)
        cuda_mod = torch_adp.get_symbol("cuda", None)
        backends_mod = torch_adp.get_symbol("backends", None)
        target = device or "cpu"
        if target == "cuda" and not bool(adapt_backend_module(cuda_mod).require_op("is_available")()):
            target = "cpu"
        elif target == "mps" and not (
            adapt_backend_module(backends_mod).has_symbol("mps")
            and bool(
                adapt_backend_module(adapt_backend_module(backends_mod).require_symbol("mps")).require_op(
                    "is_available"
                )()
            )
        ):
            target = "cpu"
        return torch_adp.require_op("device")(target)

    def _prepare_inputs(
        self,
        inputs: Mapping[str, list[float] | list[list[float]] | _torch_types.Tensor],
        device: BackendArg,
    ) -> list[BackendArg]:
        """Convert input payloads to device tensors.

        Args:
            inputs: Input dict.
            device (BackendArg): Target device.

        Returns:
            list[BackendArg]: Converted tensor list.
        """
        torch_adp = adapt_backend_module(torch)
        tensor_cls: BackendArg = torch_adp.get_symbol("Tensor", None)
        torch_inputs: list[BackendArg] = []
        for val in inputs.values():
            raw_val: BackendArg = val
            if isinstance(tensor_cls, type) and isinstance(val, tensor_cls):
                torch_inputs.append(adapt_backend_module(raw_val).require_op("to")(device))
            else:
                torch_inputs.append(
                    torch_adp.require_op("tensor")(raw_val, device=device, dtype=torch_adp.get_symbol("float32", None))
                )
        return torch_inputs

    def _prepare_ordered_inputs(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | _torch_types.Tensor],
        device: BackendArg,
    ) -> list[BackendArg]:
        """Prepare inputs ordered topologically for compiled forward execution.

        Args:
            graph (IRGraph): IR computation graph.
            inputs: Input mapping.
            device (BackendArg): Target PyTorch device.

        Returns:
            list[BackendArg]: Ordered tensor inputs list.
        """
        torch_adp = adapt_backend_module(torch)
        tensor_cls: BackendArg = torch_adp.get_symbol("Tensor", None)
        torch_inputs_dict: dict[str, BackendArg] = {}
        for k, val in inputs.items():
            raw_val: BackendArg = val
            if isinstance(tensor_cls, type) and isinstance(val, tensor_cls):
                torch_inputs_dict[k] = adapt_backend_module(raw_val).require_op("to")(device)
            else:
                torch_inputs_dict[k] = torch_adp.require_op("tensor")(
                    raw_val, device=device, dtype=torch_adp.get_symbol("float32", None)
                )

        sorted_nodes = topological_sort(graph) if graph.nodes else []
        input_keys = [n.id for n in sorted_nodes if n.op_type == "Input"]
        if input_keys and all(k in torch_inputs_dict for k in input_keys):
            return [torch_inputs_dict[k] for k in input_keys]
        return list(torch_inputs_dict.values())

    def _sync(self, device: BackendArg) -> None:
        """Synchronize execution across hardware devices.

        Args:
            device (BackendArg): Device to synchronize.
        """
        dev_type = adapt_backend_module(device).get_symbol("type", "")
        torch_adp = adapt_backend_module(torch)
        cuda_mod = torch_adp.get_symbol("cuda", None)
        if dev_type == "cuda" and bool(adapt_backend_module(cuda_mod).require_op("is_available")()):
            adapt_backend_module(cuda_mod).require_op("synchronize")()
        elif (
            dev_type == "mps"
            and torch_adp.has_symbol("mps")
            and bool(adapt_backend_module(torch_adp.require_symbol("mps")).require_op("is_available")())
        ):
            adapt_backend_module(torch_adp.require_symbol("mps")).require_op("synchronize")()

    def _reset_cuda_stats(self, device: BackendArg) -> None:
        """Reset peak memory stats on CUDA device if available.

        Args:
            device (BackendArg): Target PyTorch device.
        """
        dev_type = adapt_backend_module(device).get_symbol("type", "")
        cuda_mod = adapt_backend_module(torch).get_symbol("cuda", None)
        cuda_adp = adapt_backend_module(cuda_mod)
        if (
            dev_type == "cuda"
            and bool(cuda_adp.require_op("is_available")())
            and cuda_adp.has_symbol("reset_peak_memory_stats")
        ):
            try:
                cuda_adp.require_op("reset_peak_memory_stats")()
            except (
                AssertionError,
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
                logging.getLogger(__name__).debug("Ignored exception in %s", __name__)

    def _compile_module(
        self,
        graph: IRGraph,
        device: BackendArg,
        input_list: list[BackendArg],
    ) -> TypedVariadicCallable[BackendArg] | TorchGraphModule | None:
        """Compile an IRGraph into an optimized PyTorch module.

        Args:
            graph (IRGraph): The IR graph to compile.
            device (BackendArg): Target execution device.
            input_list (list[BackendArg]): Input tensor examples for tracing.

        Returns:
            Optional[Union[TypedVariadicCallable[BackendArg], TorchGraphModule]]: Compiled PyTorch module.
        """
        base_module = TorchGraphModule(graph).to(device)
        compiled: TypedVariadicCallable[BackendArg] | TorchGraphModule = base_module
        torch_adp = adapt_backend_module(torch)
        if callable(torch_adp.get_symbol("compile", None)):
            try:
                comp_res = torch_adp.require_op("compile")(base_module)
                bound_comp = require_callable(comp_res)
                compiled = lambda *a, **kw: bound_comp(*a, **kw)  # noqa: E731
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
                compiled = base_module
        if (
            compiled is base_module
            and torch_adp.has_symbol("jit")
            and callable(adapt_backend_module(torch_adp.require_symbol("jit")).get_symbol("trace", None))
            and input_list
        ):
            try:
                traced_res = adapt_backend_module(torch_adp.require_symbol("jit")).require_op("trace")(
                    base_module, tuple(input_list)
                )
                bound_traced = require_callable(traced_res)
                compiled = lambda *a, **kw: bound_traced(*a, **kw)  # noqa: E731
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
                compiled = base_module
        return compiled

    def _measure_latencies(
        self,
        device: BackendArg,
        step_fn: Callable[[], BackendArg],
        num_iters: int,
    ) -> list[float]:
        """Measure execution latencies over multiple iterations with device sync.

        Args:
            device (BackendArg): Execution device.
            step_fn (Callable[[], BackendArg]): Forward evaluation callback.
            num_iters (int): Measurement iteration count.

        Returns:
            list[float]: Latency measurements in milliseconds.
        """
        dev_type = adapt_backend_module(device).get_symbol("type", "")
        cuda_adp = adapt_backend_module(adapt_backend_module(torch).get_symbol("cuda", None))
        is_cuda: bool = bool(dev_type == "cuda" and cuda_adp.require_op("is_available")())
        latencies: list[float] = []
        for _ in range(max(1, num_iters)):
            cuda_timed: bool = False
            if is_cuda and cuda_adp.has_symbol("Event"):
                try:
                    event_cls = cuda_adp.require_op("Event")
                    start_event = event_cls(enable_timing=True)
                    end_event = event_cls(enable_timing=True)
                    start_event.record()
                    _ = step_fn()
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(float(coerce_float(start_event.elapsed_time(end_event))))
                    cuda_timed = True
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
                    cuda_timed = False
            if not cuda_timed:
                self._sync(device)
                start = time.perf_counter()
                _ = step_fn()
                self._sync(device)
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)
        return latencies

    def _extract_peak_memory_mb(self, device: BackendArg) -> float:
        """Extract peak device memory in megabytes with hardware-specific APIs.

        Args:
            device (BackendArg): PyTorch execution device.

        Returns:
            float: Peak memory in megabytes.
        """
        dev_type = adapt_backend_module(device).get_symbol("type", "")
        torch_adp = adapt_backend_module(torch)
        cuda_adp = adapt_backend_module(torch_adp.get_symbol("cuda", None))
        if dev_type == "cuda" and bool(cuda_adp.require_op("is_available")()):
            return float(coerce_float(cuda_adp.require_op("max_memory_allocated")()) / (1024.0 * 1024.0))
        if dev_type == "mps":
            mps_mod = torch_adp.get_symbol("mps", None)
            if mps_mod is not None and adapt_backend_module(mps_mod).has_symbol("current_allocated_memory"):
                try:
                    mps_mem = float(
                        coerce_float(adapt_backend_module(mps_mod).require_op("current_allocated_memory")())
                    ) / (1024.0 * 1024.0)
                    if mps_mem > 0.0:
                        return mps_mem
                except (ImportError, AttributeError, TypeError, ValueError, KeyError, RuntimeError, OSError):
                    logging.getLogger(__name__).debug("Ignored exception in %s", __name__)
        return _get_process_memory_mb()

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: Mapping[str, list[float] | list[list[float]] | _torch_types.Tensor],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfileMetricsDict:
        """Profile a graph execution on the PyTorch backend with hardware synchronization.

        Args:
            graph (IRGraph): The IR computation graph to execute.
            inputs: Named input tensors or buffers.
            device (Optional[str]): Device target ('cpu', 'cuda', 'mps').
            num_iters (int): Measurement iteration count.
            warmup_iters (int): Warmup iteration count.

        Returns:
            ProfileMetricsDict: Latency and memory metrics.
        """
        torch_dev = self._resolve_device(device)
        has_ops: bool = bool(graph.nodes and any(n.op_type != "Input" for n in graph.nodes.values()))
        torch_input_list: list[BackendArg] = self._prepare_ordered_inputs(graph, inputs, torch_dev)

        compiled_module: TypedVariadicCallable[BackendArg] | TorchGraphModule | None = None
        if has_ops:
            compiled_module = self._compile_module(graph, torch_dev, torch_input_list)

        def _execute_step() -> BackendArg:
            """Execute a single evaluation step on PyTorch backend.

            Returns:
                BackendArg: Single or collection of output tensors.
            """
            if has_ops and compiled_module is not None:
                return compiled_module(*torch_input_list)
            torch_step = adapt_backend_module(torch)
            if not torch_input_list:
                return torch_step.require_op("zeros")(
                    (1,), device=torch_dev, dtype=torch_step.get_symbol("float32", None)
                )
            accum: BackendArg = torch_step.require_op("zeros_like")(torch_input_list[0])
            for t in torch_input_list:
                accum = torch_step.require_op("add")(accum, t)
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
