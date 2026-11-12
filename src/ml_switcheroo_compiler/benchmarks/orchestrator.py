"""Runtime orchestrator for cross-backend benchmarking and performance verification."""

from __future__ import annotations

import importlib
import inspect
import multiprocessing
import pathlib
import typing
from collections.abc import Sequence

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.benchmarks.config_models import (
    BenchmarkPlan,
    BenchmarkRunResult,
    ProfilerProfileModel,
    load_backend_profiles,
)
from ml_switcheroo_compiler.benchmarks.metrics import compute_statistical_metrics
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.interpreter.evaluator import EvalValue
from ml_switcheroo_compiler.ir.core import IRGraph

ProfilerMetrics = dict[str, typing.Union[list[float], "float"]]
BenchmarkResult = tuple[str, typing.Union[ProfilerMetrics, "str"]]


def _to_metric_float(val: float | list[float] | int | str | None) -> float:
    """Coerce a statistical metric value to float.

    Args:
        val (float | list[float] | int | str | None): Metric value.

    Returns:
        float: Coerced float value.
    """
    return float(val) if isinstance(val, (int, float)) else 0.0


@typing.runtime_checkable
class _ProfilerFactoryProtocol(typing.Protocol):
    """Protocol for zero-argument profiler class constructors."""

    def __call__(self) -> _ProfilerProtocol:
        """Instantiate profiler.

        Returns:
            _ProfilerProtocol: Profiler instance.
        """
        ...


@typing.runtime_checkable
class _ProfilerProtocol(typing.Protocol):
    """Internal protocol defining the profiling execution signature."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, EvalValue],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfilerMetrics:
        """Profile a graph execution.

        Args:
            graph: Input `graph` argument (graph).
            inputs: Input `inputs` argument (inputs).
            device: Input `device` argument (device).
            num_iters: Input `num_iters` argument (num iters).
            warmup_iters: Input `warmup_iters` argument (warmup iters).

        Returns:
            ProfilerMetrics: Computed result.
        """
        ...


@typing.runtime_checkable
class _ExecuteOpBackendProtocol(typing.Protocol):
    """Protocol for backends exposing `execute_op`."""

    def execute_op(self, op_name: str, *args: int | EvalValue, **kwargs: str) -> EvalValue:
        """Execute a backend operation by name.

        Args:
            op_name (str): Operation name.
            *args (int | EvalValue): Positional arguments.
            **kwargs (str): Keyword arguments.

        Returns:
            EvalValue: Operation output.
        """
        ...


def _is_profiler_factory(val: TensorOrScalar) -> typing.TypeGuard[_ProfilerFactoryProtocol]:
    """Type guard narrowing `val` to `_ProfilerFactoryProtocol`.

    Args:
        val (TensorOrScalar): Candidate profiler class.

    Returns:
        bool: True if callable.
    """
    return callable(val)


def _instantiate_profiler(mod_name: str, cls_name: str) -> _ProfilerProtocol:
    """Import `mod_name` and instantiate profiler class `cls_name`.

    Args:
        mod_name (str): Module path.
        cls_name (str): Class name inside `mod_name`.

    Returns:
        _ProfilerProtocol: Instantiated profiler.
    """
    mod = importlib.import_module(mod_name)
    members = dict(inspect.getmembers(mod))
    profiler_cls: TensorOrScalar = members[cls_name]
    assert _is_profiler_factory(profiler_cls)
    return profiler_cls()


def _isolated_worker(
    profiler_cls_tuple: tuple[str, str],
    graph: IRGraph,
    inputs: dict[str, EvalValue],
    device: str | None,
    num_iters: int,
    warmup_iters: int,
    output_queue: multiprocessing.Queue[BenchmarkResult],
) -> None:
    """Execute profiling in an isolated process to prevent namespace pollution.

    Args:
        profiler_cls_tuple (tuple[str, str]): Module and class names for profiler.
        graph (IRGraph): Computational graph.
        inputs (dict[str, EvalValue]): Input arrays.
        device (Optional[str]): Device identifier.
        num_iters (int): Measurement iteration count.
        warmup_iters (int): Warmup iteration count.
        output_queue (multiprocessing.Queue): Queue to communicate results back.
    """
    try:
        profiler = _instantiate_profiler(profiler_cls_tuple[0], profiler_cls_tuple[1])
        metrics: ProfilerMetrics = profiler.profile_graph(
            graph=graph,
            inputs=inputs,
            device=device,
            num_iters=num_iters,
            warmup_iters=warmup_iters,
        )
        output_queue.put(("ok", metrics))
    except Exception as exc:
        output_queue.put(("error", str(exc)))


class BenchmarkOrchestrator:
    """Orchestrates cross-backend compilation, execution profiling, and benchmarking."""

    def __init__(self, plan: BenchmarkPlan) -> None:
        """Initialize the benchmark orchestrator with a declarative plan.

        Args:
            plan (BenchmarkPlan): The declarative benchmark plan.
        """
        self.plan: BenchmarkPlan = plan

    @classmethod
    def _load_manifest_profiles(cls) -> dict[str, tuple[str, str]]:
        """Load declarative profiler configurations from backend profiles manifest.

        Returns:
            dict[str, tuple[str, str]]: Mapping from backend name to (module, class) profiler specs.
        """
        manifest_path = pathlib.Path(__file__).parent / "manifests" / "backend_profiles.yaml"
        if not manifest_path.exists():
            return {}
        try:
            profiles = load_backend_profiles(str(manifest_path))
            specs: dict[str, tuple[str, str]] = {}
            for bname, profile in profiles.items():
                if profile.profiler_module and profile.profiler_class:
                    specs[str(bname)] = (str(profile.profiler_module), str(profile.profiler_class))
            return specs
        except Exception:
            return {}

    def _get_profiler(self, backend_name: str) -> _ProfilerProtocol:
        """Retrieve the dedicated execution profiler for a given backend.

        Args:
            backend_name (str): The name of the backend (e.g. 'numpy', 'pytorch', 'jax', 'mlx').

        Returns:
            _ProfilerProtocol: Instantiated backend profiler.

        Raises:
            BackendNotSupportedError: If backend has no supported profiler or import fails.
        """
        canonical: str = "pytorch" if backend_name in ("torch", "pytorch") else backend_name
        profiles = load_backend_profiles()
        profile: ProfilerProfileModel | None = profiles.get(canonical.lower()) or profiles.get(backend_name.lower())

        if profile is not None and profile.profiler_module and profile.profiler_class:
            try:
                return _instantiate_profiler(profile.profiler_module, profile.profiler_class)
            except ImportError as err:
                raise BackendNotSupportedError(f"Profiler for backend '{backend_name}' could not be imported: {err}") from err

        raise BackendNotSupportedError(f"Backend '{backend_name}' does not have a supported profiler.")

    def _run_isolated(
        self,
        graph: IRGraph,
        backend_name: str,
        inputs: dict[str, EvalValue],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfilerMetrics:
        """Execute profiling in an isolated process to prevent interpreter state contamination.

        Args:
            graph (IRGraph): The computation graph.
            backend_name (str): Target backend name.
            inputs (dict[str, EvalValue]): Inputs mapping.
            device (Optional[str]): Device identifier.
            num_iters (int): Measurement iterations.
            warmup_iters (int): Warmup iterations.

        Returns:
            ProfilerMetrics: Collected metrics.

        Raises:
            RuntimeError: If isolated execution fails or times out.
        """
        canonical = "pytorch" if backend_name in ("torch", "pytorch") else backend_name
        profiles = load_backend_profiles()
        profile = profiles.get(canonical.lower()) or profiles.get(backend_name.lower())
        if profile is not None and profile.profiler_module and profile.profiler_class:
            spec = (profile.profiler_module, profile.profiler_class)
        else:
            spec = ("ml_switcheroo_compiler.backends.numpy.profiler", "NumpyProfiler")

        ctx = multiprocessing.get_context()
        queue: multiprocessing.Queue[BenchmarkResult] = ctx.Queue()
        proc = ctx.Process(
            target=_isolated_worker,
            args=(spec, graph, inputs, device, num_iters, warmup_iters, queue),
        )
        proc.start()
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise RuntimeError(f"Isolated benchmark run timed out for backend '{backend_name}'.")

        if queue.empty():
            raise RuntimeError(f"Isolated benchmark run for backend '{backend_name}' terminated unexpectedly.")

        status, payload = queue.get()
        if status != "ok" or not isinstance(payload, dict):
            raise RuntimeError(f"Isolated execution failed: {payload}")
        return payload

    def _run_single(
        self,
        graph: IRGraph,
        backend_name: str,
        batch_size: int,
        num_iters: int,
        warmup_iters: int,
        device: str | None = None,
        isolate: bool | None = None,
    ) -> ProfilerMetrics:
        """Run a single benchmark iteration measuring latency and memory.

        Args:
            graph (IRGraph): The computation graph.
            backend_name (str): The name of the target backend.
            batch_size (int): Batch size used for inputs.
            num_iters (int): Number of measurement iterations.
            warmup_iters (int): Number of warmup iterations.
            device (Optional[str]): Device identifier.
            isolate (Optional[bool]): Whether to isolate execution in a subprocess.

        Returns:
            ProfilerMetrics: Dictionary containing latencies and peak memory usage.

        Raises:
            BackendNotSupportedError: If backend is not registered.
        """
        # Validate backend registration
        try:
            _ = BackendRegistry.get(backend_name)
        except (ValueError, KeyError) as exc:
            raise BackendNotSupportedError(f"Backend '{backend_name}' is not registered.") from exc

        # Construct inputs from plan or graph declarations
        from ml_switcheroo_compiler.backends.registry import get_active_backend

        inputs_dict: dict[str, EvalValue] = {}
        input_names: list[str] = list(graph.inputs) if graph.inputs else ["in_0"]

        active_backend: TensorOrScalar = get_active_backend()
        for inp in input_names:
            node = graph.nodes.get(inp) if graph.nodes else None
            node_attrs = (node.attributes or {}) if node is not None else {}
            raw_shape = node_attrs.get("shape")
            node_shape: list[int] = [int(d) for d in raw_shape if isinstance(d, int)] if isinstance(raw_shape, (list, tuple, Sequence)) and not isinstance(raw_shape, (str, bytes)) else []
            is_weight = bool(node_attrs.get("is_weight", False))

            if is_weight and node_shape:
                full_shape: tuple[int, ...] = tuple(node_shape)
            else:
                extra_dims: list[int] = self.plan.input_shapes.get(inp, node_shape if node_shape else [10])
                full_shape = (batch_size, *extra_dims)
            if isinstance(active_backend, _ExecuteOpBackendProtocol):
                inputs_dict[inp] = active_backend.execute_op("Cast", active_backend.execute_op("Randn", *full_shape), dtype="float32")
            else:
                inputs_dict[inp] = [0.0]

        should_isolate: bool = isolate if isolate is not None else bool(self.plan.isolate_process)
        if should_isolate:
            return self._run_isolated(
                graph=graph,
                backend_name=backend_name,
                inputs=inputs_dict,
                device=device,
                num_iters=num_iters,
                warmup_iters=warmup_iters,
            )

        profiler: _ProfilerProtocol = self._get_profiler(backend_name)
        return profiler.profile_graph(
            graph=graph,
            inputs=inputs_dict,
            device=device,
            num_iters=num_iters,
            warmup_iters=warmup_iters,
        )

    def execute(self) -> list[BenchmarkRunResult]:
        """Execute the declarative benchmark plan across all configured targets.

        Returns:
            list[BenchmarkRunResult]: Collected metrics for all runs.
        """
        results: list[BenchmarkRunResult] = []
        from ml_switcheroo_compiler.benchmarks.config_models import build_ir_graph_from_workload, load_model_workloads

        try:
            workload_manifest = load_model_workloads()
        except Exception:
            workload_manifest = {}

        for model_name in self.plan.models:
            if model_name in workload_manifest:
                target_graph: IRGraph = build_ir_graph_from_workload(workload_manifest[model_name], name=model_name)
            else:
                target_graph = IRGraph(name=model_name)
                target_graph.inputs = ["in_0"]

            for batch_size in self.plan.batch_sizes:
                for target in self.plan.targets:
                    try:
                        metrics: ProfilerMetrics = self._run_single(
                            graph=target_graph,
                            backend_name=target.backend,
                            batch_size=batch_size,
                            num_iters=self.plan.num_iterations,
                            warmup_iters=self.plan.warmup_iterations,
                            device=target.device,
                        )
                    except TypeError:
                        metrics = self._run_single(
                            graph=target_graph,
                            backend_name=target.backend,
                            batch_size=batch_size,
                            num_iters=self.plan.num_iterations,
                            warmup_iters=self.plan.warmup_iterations,
                        )

                    raw_latencies = metrics.get("latencies", [])
                    lat_list: list[float] = [float(x) for x in raw_latencies] if isinstance(raw_latencies, list) and raw_latencies else [0.0]
                    stat_metrics = compute_statistical_metrics(lat_list)
                    mean_lat: float = _to_metric_float(stat_metrics["mean_ms"])

                    peak_mem_raw = metrics.get("peak_memory_mb")
                    peak_mem: float | None = float(peak_mem_raw) if isinstance(peak_mem_raw, (int, float)) else None

                    p50: float = _to_metric_float(stat_metrics["p50_ms"])
                    p95: float = _to_metric_float(stat_metrics["p95_ms"])
                    p99: float = _to_metric_float(stat_metrics["p99_ms"])

                    throughput: float = (1000.0 / mean_lat) * batch_size if mean_lat > 0.0 else 0.0

                    res = BenchmarkRunResult(
                        model=model_name,
                        batch_size=batch_size,
                        backend=target.backend,
                        device=target.device or "cpu",
                        mean_latency_ms=mean_lat,
                        p50_latency_ms=p50,
                        p95_latency_ms=p95,
                        p99_latency_ms=p99,
                        peak_memory_mb=peak_mem,
                        throughput_items_per_sec=throughput,
                    )
                    results.append(res)

        return results
