"""Tests for the benchmarking orchestrator."""

from typing import Any

from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkPlan, BenchmarkRunResult, BenchmarkTarget
from ml_switcheroo_compiler.benchmarks.orchestrator import BenchmarkOrchestrator
from ml_switcheroo_compiler.ir.core import IRGraph


def test_benchmark_plan_model():
    """Test the benchmark plan data model."""
    target = BenchmarkTarget(backend="numpy", device="cpu")
    plan = BenchmarkPlan(name="test_plan", models=["resnet18"], batch_sizes=[1, 32], targets=[target], num_iterations=10, warmup_iterations=2)
    assert plan.name == "test_plan"
    assert len(plan.targets) == 1
    assert plan.targets[0].backend == "numpy"


def test_orchestrator_execution(monkeypatch):
    """Test the orchestrator execution loop."""

    # Mock the backend registry to avoid requiring real backends
    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend, raising=False)

    target = BenchmarkTarget(backend="numpy", device="cpu")
    plan = BenchmarkPlan(name="test_plan", models=["resnet18"], batch_sizes=[1], targets=[target], num_iterations=5, warmup_iterations=1)

    orchestrator = BenchmarkOrchestrator(plan)

    # We mock _run_single to avoid time.sleep delays in tests
    original_run_single = orchestrator._run_single

    def fast_run_single(graph: IRGraph, backend_name: str, batch_size: int, num_iters: int, warmup_iters: int) -> dict[str, Any]:
        return {"latencies": [10.0] * num_iters, "peak_memory_mb": 50.0}

    monkeypatch.setattr(orchestrator, "_run_single", fast_run_single)

    results = orchestrator.execute()

    assert len(results) == 1
    res = results[0]
    assert isinstance(res, BenchmarkRunResult)
    assert res.model == "resnet18"
    assert res.batch_size == 1
    assert res.backend == "numpy"
    assert res.mean_latency_ms == 10.0
    assert res.peak_memory_mb == 50.0
    assert res.throughput_items_per_sec == 100.0  # (1000 / 10) * 1


def test_orchestrator_internal_run(monkeypatch):
    """Test the _run_single internal logic explicitly."""

    # We patch time.sleep to avoid waiting
    def mock_sleep(secs: float) -> None:
        pass

    monkeypatch.setattr("time.sleep", mock_sleep)

    # Mock time.perf_counter to return deterministic increments
    counter = [0.0]

    def mock_perf_counter() -> float:
        val = counter[0]
        counter[0] += 0.005  # Simulate 5ms elapsed
        return val

    monkeypatch.setattr("time.perf_counter", mock_perf_counter)

    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend, raising=False)

    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="dummy")])
    orchestrator = BenchmarkOrchestrator(plan)

    metrics = orchestrator._run_single(IRGraph(), "dummy", 1, 5, 2)
    assert len(metrics["latencies"]) == 5
    assert metrics["latencies"][0] == 5.0  # 0.005 seconds * 1000 = 5.0ms


def test_orchestrator_zero_latency(monkeypatch):
    """Test the throughput_items_per_sec with 0 latency."""
    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="dummy")])
    orchestrator = BenchmarkOrchestrator(plan)

    def fast_run_single(graph: IRGraph, backend_name: str, batch_size: int, num_iters: int, warmup_iters: int) -> dict[str, Any]:
        return {"latencies": [0.0] * num_iters, "peak_memory_mb": 50.0}

    monkeypatch.setattr(orchestrator, "_run_single", fast_run_single)
    results = orchestrator.execute()
    assert results[0].throughput_items_per_sec == 0.0


def test_orchestrator_internal_run_with_inputs(monkeypatch):
    """Test the _run_single internal logic explicitly with inputs."""

    # We patch time.sleep to avoid waiting
    def mock_sleep(secs: float) -> None:
        pass

    monkeypatch.setattr("time.sleep", mock_sleep)

    # Mock time.perf_counter to return deterministic increments
    counter = [0.0]

    def mock_perf_counter() -> float:
        val = counter[0]
        counter[0] += 0.005  # Simulate 5ms elapsed
        return val

    monkeypatch.setattr("time.perf_counter", mock_perf_counter)

    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend)

    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="dummy")])
    orchestrator = BenchmarkOrchestrator(plan)

    g = IRGraph()
    g.inputs = ["in_1"]
    metrics = orchestrator._run_single(g, "dummy", 1, 1, 1)
    assert len(metrics["latencies"]) == 1


def test_mlx_profiler():
    """Test mlx profiler mock."""
    from ml_switcheroo_compiler.backends.mlx.profiler import MLXProfiler

    res = MLXProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res


def test_numpy_profiler():
    """Test numpy profiler mock."""
    from ml_switcheroo_compiler.backends.numpy.profiler import NumpyProfiler

    res = NumpyProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res
