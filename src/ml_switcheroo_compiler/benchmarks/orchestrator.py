"""Runtime orchestrator for cross-backend benchmarking."""

import time
from typing import Any

import numpy as np

from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkPlan, BenchmarkRunResult
from ml_switcheroo_compiler.ir.core import IRGraph


class BenchmarkOrchestrator:
    """Orchestrates cross-backend compilation and benchmarking."""

    def __init__(self, plan: BenchmarkPlan) -> None:
        """Initialize orchestrator with a plan.

        Args:
            plan (BenchmarkPlan): The declarative benchmark plan.
        """
        self.plan = plan

    def _run_single(self, graph: IRGraph, backend_name: str, batch_size: int, num_iters: int, warmup_iters: int) -> dict[str, Any]:
        """Run a single benchmark iteration.

        Args:
            graph (IRGraph): The computation graph.
            backend_name (str): The name of the target backend.
            batch_size (int): Batch size used for inputs.
            num_iters (int): Number of measurement iterations.
            warmup_iters (int): Number of warmup iterations.

        Returns:
            dict[str, Any]: Dictionary containing latencies and memory usage.
        """
        # Instantiate backend execution context (this maps to whatever eager API the backend uses)
        # Note: In a real implementation this would invoke the specific profiler for the backend
        _ = BackendRegistry.get(backend_name)

        # Construct dummy inputs (using numpy, which is allowed)
        inputs_dict = {}
        for inp in getattr(graph, "inputs", []):
            inputs_dict[inp] = np.random.randn(batch_size, 10).astype(np.float32)  # Simplified shape

        # Warmup
        for _ in range(warmup_iters):
            # Since we abstract execution, we'll just mock the timing for this skeleton
            # Real implementation would call backend_api.execute(graph, inputs_dict)
            time.sleep(0.001)

        latencies = []
        for _ in range(num_iters):
            start = time.perf_counter()
            # backend_api.execute(graph, inputs_dict)
            time.sleep(0.005)  # mock
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

        return {
            "latencies": latencies,
            "peak_memory_mb": 100.0,  # Mock peak memory
        }

    def execute(self) -> list[BenchmarkRunResult]:
        """Execute the benchmark plan across all backends.

        Returns:
            list[BenchmarkRunResult]: Collected metrics for all runs.
        """
        results = []

        for model_name in self.plan.models:
            # We would normally load the graph here
            dummy_graph = IRGraph()
            dummy_graph.inputs = ["in_0"]

            for batch_size in self.plan.batch_sizes:
                for target in self.plan.targets:
                    metrics = self._run_single(dummy_graph, target.backend, batch_size, self.plan.num_iterations, self.plan.warmup_iterations)

                    latencies = np.array(metrics["latencies"])
                    mean_lat = float(np.mean(latencies))

                    res = BenchmarkRunResult(
                        model=model_name,
                        batch_size=batch_size,
                        backend=target.backend,
                        device=target.device or "cpu",
                        mean_latency_ms=mean_lat,
                        p50_latency_ms=float(np.percentile(latencies, 50)),
                        p95_latency_ms=float(np.percentile(latencies, 95)),
                        p99_latency_ms=float(np.percentile(latencies, 99)),
                        peak_memory_mb=metrics["peak_memory_mb"],
                        throughput_items_per_sec=(1000.0 / mean_lat) * batch_size if mean_lat > 0 else 0.0,
                    )
                    results.append(res)

        return results
