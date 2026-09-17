"""Runtime orchestrator for cross-backend benchmarking and performance verification."""

import importlib
import multiprocessing
import pathlib
from typing import Optional, Protocol, Union

import numpy as np

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.benchmarks.config_models import (
    BenchmarkPlan,
    BenchmarkRunResult,
    ProfilerProfileModel,
    load_backend_profiles,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph

ProfilerMetrics = dict[str, Union[list[float], float]]


class _ProfilerProtocol(Protocol):
    """Internal protocol defining the profiling execution signature."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfilerMetrics:
        """Profile a graph execution."""
        ...


def _isolated_worker(
    profiler_cls_tuple: tuple[str, str],
    graph: IRGraph,
    inputs: dict[str, np.ndarray],
    device: Optional[str],
    num_iters: int,
    warmup_iters: int,
    output_queue: multiprocessing.Queue,
) -> None:
    """Execute profiling in an isolated process to prevent namespace pollution.

    Args:
        profiler_cls_tuple (tuple[str, str]): Module and class names for profiler.
        graph (IRGraph): Computational graph.
        inputs (dict[str, np.ndarray]): Input arrays.
        device (Optional[str]): Device identifier.
        num_iters (int): Measurement iteration count.
        warmup_iters (int): Warmup iteration count.
        output_queue (multiprocessing.Queue): Queue to communicate results back.
    """
    try:
        mod = importlib.import_module(profiler_cls_tuple[0])
        profiler_cls = getattr(mod, profiler_cls_tuple[1])
        profiler = profiler_cls()
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
        profile: Optional[ProfilerProfileModel] = profiles.get(canonical.lower()) or profiles.get(backend_name.lower())

        if profile is not None and profile.profiler_module and profile.profiler_class:
            try:
                mod = importlib.import_module(profile.profiler_module)
                profiler_cls = getattr(mod, profile.profiler_class)
                return profiler_cls()
            except ImportError as err:
                raise BackendNotSupportedError(f"Profiler for backend '{backend_name}' could not be imported: {err}") from err

        raise BackendNotSupportedError(f"Backend '{backend_name}' does not have a supported profiler.")

    def _run_isolated(
        self,
        graph: IRGraph,
        backend_name: str,
        inputs: dict[str, np.ndarray],
        device: Optional[str] = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> ProfilerMetrics:
        """Execute profiling in an isolated process to prevent interpreter state contamination.

        Args:
            graph (IRGraph): The computation graph.
            backend_name (str): Target backend name.
            inputs (dict[str, np.ndarray]): Inputs mapping.
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
        queue: multiprocessing.Queue = ctx.Queue()
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
        if status != "ok":
            raise RuntimeError(f"Isolated execution failed: {payload}")
        return payload

    def _run_single(
        self,
        graph: IRGraph,
        backend_name: str,
        batch_size: int,
        num_iters: int,
        warmup_iters: int,
        device: Optional[str] = None,
        isolate: Optional[bool] = None,
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
        inputs_dict: dict[str, np.ndarray] = {}
        input_names: list[str] = list(getattr(graph, "inputs", []))
        if not input_names:
            input_names = ["in_0"]

        for inp in input_names:
            node = graph.nodes.get(inp) if getattr(graph, "nodes", None) else None
            node_attrs = getattr(node, "attributes", {}) or {}
            node_shape = node_attrs.get("shape")
            is_weight = bool(node_attrs.get("is_weight", False))

            if is_weight and node_shape:
                full_shape: tuple[int, ...] = tuple(node_shape)
            else:
                extra_dims: list[int] = self.plan.input_shapes.get(inp, node_shape if node_shape else [10])
                full_shape = (batch_size, *extra_dims)
            inputs_dict[inp] = np.random.randn(*full_shape).astype(np.float32)

        should_isolate: bool = isolate if isolate is not None else getattr(self.plan, "isolate_process", False)
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
                    latencies: np.ndarray = np.array(raw_latencies if isinstance(raw_latencies, list) else [0.0])
                    mean_lat: float = float(np.mean(latencies)) if latencies.size > 0 else 0.0

                    peak_mem_raw = metrics.get("peak_memory_mb")
                    peak_mem: Optional[float] = float(peak_mem_raw) if isinstance(peak_mem_raw, (int, float)) else None

                    p50: float = float(np.percentile(latencies, 50)) if latencies.size > 0 else 0.0
                    p95: float = float(np.percentile(latencies, 95)) if latencies.size > 0 else 0.0
                    p99: float = float(np.percentile(latencies, 99)) if latencies.size > 0 else 0.0

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
