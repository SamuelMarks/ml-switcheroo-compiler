"""Tests for the benchmarking orchestrator."""

import os
import tempfile
from typing import Any

import numpy as np
import pytest

from ml_switcheroo_compiler.benchmarks.config_models import (
    BenchmarkPlan,
    BenchmarkRunResult,
    BenchmarkSuite,
    BenchmarkTarget,
    load_benchmark_plan,
    load_benchmark_suite,
)
from ml_switcheroo_compiler.benchmarks.orchestrator import BenchmarkOrchestrator
from ml_switcheroo_compiler.ir.core import IRGraph


def test_benchmark_plan_model():
    """Test the benchmark plan data model."""
    target = BenchmarkTarget(backend="numpy", device="cpu")
    plan = BenchmarkPlan(
        name="test_plan",
        models=["resnet18"],
        batch_sizes=[1, 32],
        targets=[target],
        num_iterations=10,
        warmup_iterations=2,
    )
    assert plan.name == "test_plan"
    assert len(plan.targets) == 1
    assert plan.targets[0].backend == "numpy"


def test_orchestrator_execution(monkeypatch):
    """Test the orchestrator execution loop."""

    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend, raising=False)

    target = BenchmarkTarget(backend="numpy", device="cpu")
    plan = BenchmarkPlan(
        name="test_plan",
        models=["resnet18"],
        batch_sizes=[1],
        targets=[target],
        num_iterations=5,
        warmup_iterations=1,
    )

    orchestrator = BenchmarkOrchestrator(plan)

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

    def mock_sleep(secs: float) -> None:
        pass

    monkeypatch.setattr("time.sleep", mock_sleep)

    counter = [0.0]

    def mock_perf_counter() -> float:
        val = counter[0]
        counter[0] += 0.005
        return val

    monkeypatch.setattr("time.perf_counter", mock_perf_counter)

    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend, raising=False)

    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="numpy")])
    orchestrator = BenchmarkOrchestrator(plan)

    metrics = orchestrator._run_single(IRGraph(), "numpy", 1, 5, 2)
    assert len(metrics["latencies"]) == 5
    assert metrics["latencies"][0] == 5.0


def test_orchestrator_unregistered_backend():
    """Test _run_single with an unregistered backend name to trigger BackendNotSupportedError in registry lookup."""
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="non_existent_backend")])
    orchestrator = BenchmarkOrchestrator(plan)
    with pytest.raises(BackendNotSupportedError):
        orchestrator._run_single(IRGraph(), "non_existent_backend", 1, 2, 1)


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

    def mock_sleep(secs: float) -> None:
        pass

    monkeypatch.setattr("time.sleep", mock_sleep)

    counter = [0.0]

    def mock_perf_counter() -> float:
        val = counter[0]
        counter[0] += 0.005
        return val

    monkeypatch.setattr("time.perf_counter", mock_perf_counter)

    def mock_get_backend(name: str):
        class MockBackend:
            pass

        return MockBackend()

    monkeypatch.setattr("ml_switcheroo_compiler.benchmarks.orchestrator.BackendRegistry.get", mock_get_backend)

    plan = BenchmarkPlan(name="test", models=["m1"], batch_sizes=[1], targets=[BenchmarkTarget(backend="numpy")])
    orchestrator = BenchmarkOrchestrator(plan)

    g = IRGraph()
    g.inputs = ["in_1"]
    metrics = orchestrator._run_single(g, "numpy", 1, 1, 1)
    assert len(metrics["latencies"]) == 1


def test_mlx_profiler():
    """Test mlx profiler execution."""
    from ml_switcheroo_compiler.backends.mlx.profiler import MLXProfiler

    res_empty = MLXProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res_empty
    assert "peak_memory_mb" in res_empty

    inp = {"x": [1.0, 2.0, 3.0]}
    res = MLXProfiler().profile_graph(IRGraph(), inp, num_iters=3, warmup_iters=1)
    assert res["latency_ms"] >= 0.0
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0


def test_numpy_profiler():
    """Test numpy profiler execution."""
    from ml_switcheroo_compiler.backends.numpy.profiler import NumpyProfiler

    res_empty = NumpyProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res_empty
    assert "peak_memory_mb" in res_empty

    inp = {"x": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    res = NumpyProfiler().profile_graph(IRGraph(), inp, num_iters=3, warmup_iters=1)
    assert res["latency_ms"] >= 0.0
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0


def test_pytorch_profiler():
    """Test pytorch profiler execution on CPU."""
    from ml_switcheroo_compiler.backends.pytorch.profiler import PyTorchProfiler

    res_empty = PyTorchProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res_empty

    inp = {"x": [1.0, 2.0, 3.0]}
    res = PyTorchProfiler().profile_graph(IRGraph(), inp, device="cpu", num_iters=3, warmup_iters=1)
    assert res["latency_ms"] >= 0.0
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0


def test_jax_profiler():
    """Test jax profiler execution."""
    from ml_switcheroo_compiler.backends.jax.profiler import JAXProfiler

    res_empty = JAXProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res_empty

    inp = {"x": [1.0, 2.0, 3.0]}
    res = JAXProfiler().profile_graph(IRGraph(), inp, num_iters=3, warmup_iters=1)
    assert res["latency_ms"] >= 0.0
    assert len(res["latencies"]) == 3
    assert res["peak_memory_mb"] > 0.0


def test_cupy_profiler():
    """Test cupy profiler execution."""
    from ml_switcheroo_compiler.backends.cupy.profiler import CupyProfiler

    res_empty = CupyProfiler().profile_graph(IRGraph(), {})
    assert "latency_ms" in res_empty


def test_orchestrator_profiler_import_error_fallback(monkeypatch):
    """Test profiler error raising when module import raises ImportError."""
    import importlib

    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    orig_import = importlib.import_module

    def mock_import(name, package=None):
        if "ml_switcheroo_compiler.backends.pytorch.profiler" in name:
            raise ImportError("Simulated missing profiler")
        return orig_import(name, package)

    monkeypatch.setattr(importlib, "import_module", mock_import)
    orchestrator = BenchmarkOrchestrator(BenchmarkPlan(name="t", models=["m"], batch_sizes=[1], targets=[]))
    with pytest.raises(BackendNotSupportedError, match="could not be imported"):
        orchestrator._get_profiler("pytorch")

    with pytest.raises(BackendNotSupportedError, match="does not have a supported profiler"):
        orchestrator._get_profiler("unsupported_backend_xyz")


def test_load_benchmark_yaml():
    """Test loading benchmark suite and plan from YAML."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "benchmarks", "benchmarks.yaml")
    suite = load_benchmark_suite(yaml_path)
    assert isinstance(suite, BenchmarkSuite)
    assert len(suite.benchmarks) > 0

    plan = suite.benchmarks[0]
    assert plan.name == "mlp_cross_backend"
    assert "numpy" in [t.backend for t in plan.targets]

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("name: single_plan\nmodels: ['m1']\nbatch_sizes: [1]\ntargets: [{backend: 'numpy'}]\n")
        temp_path = f.name

    try:
        loaded_plan = load_benchmark_plan(temp_path)
        assert loaded_plan.name == "single_plan"
    finally:
        os.unlink(temp_path)


def test_load_benchmark_yaml_errors():
    """Test error handling in benchmark YAML loaders."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_plan("non_existent_file.yaml")

    with pytest.raises(FileNotFoundError):
        load_benchmark_suite("non_existent_file.yaml")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("not_a_dict")
        temp_path = f.name

    try:
        with pytest.raises(ValueError):
            load_benchmark_plan(temp_path)
        with pytest.raises(ValueError):
            load_benchmark_suite(temp_path)
    finally:
        os.unlink(temp_path)


def test_real_cross_backend_execution():
    """Test genuine end-to-end benchmark run across NumPy, PyTorch, JAX, and MLX."""
    targets = [
        BenchmarkTarget(backend="numpy", device="cpu"),
        BenchmarkTarget(backend="pytorch", device="cpu"),
        BenchmarkTarget(backend="jax", device="cpu"),
        BenchmarkTarget(backend="mlx", device="cpu"),
    ]
    plan = BenchmarkPlan(
        name="cross_backend_eval",
        models=["linear_layer"],
        batch_sizes=[4, 8],
        targets=targets,
        num_iterations=3,
        warmup_iterations=1,
        input_shapes={"in_0": [16]},
    )

    orchestrator = BenchmarkOrchestrator(plan)
    results = orchestrator.execute()

    assert len(results) == len(targets) * len(plan.batch_sizes)
    for res in results:
        assert res.mean_latency_ms >= 0.0
        assert res.peak_memory_mb is not None
        assert res.peak_memory_mb > 0.0
        assert res.throughput_items_per_sec >= 0.0


def test_numerical_equivalence_across_backends():
    """Verify numerical equivalence across backends on identical operations."""
    import jax.numpy as jnp
    import mlx.core as mx
    import torch

    raw_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    # 1. NumPy
    np_res = np.add(raw_data, raw_data)

    # 2. PyTorch
    torch_t = torch.tensor(raw_data)
    pt_res = torch.add(torch_t, torch_t).numpy()

    # 3. JAX
    jax_t = jnp.asarray(raw_data)
    jax_res = np.asarray(jnp.add(jax_t, jax_t))

    # 4. MLX
    mlx_t = mx.array(raw_data)
    mlx_res = np.asarray(mx.add(mlx_t, mlx_t))

    np.testing.assert_allclose(np_res, pt_res, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np_res, jax_res, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np_res, mlx_res, rtol=1e-5, atol=1e-5)


def test_orchestrator_standard_ml_workloads_percentiles_and_memory():
    """Test standard ML architecture workloads asserting latency percentiles and memory tracking."""
    standard_models = ["mlp", "cnn_resnet_block", "nanogpt_transformer_block", "diffusion_resblock"]
    target = BenchmarkTarget(backend="numpy", device="cpu")
    plan = BenchmarkPlan(
        name="standard_ml_architectures_bench",
        models=standard_models,
        batch_sizes=[2],
        targets=[target],
        num_iterations=5,
        warmup_iterations=2,
    )

    orchestrator = BenchmarkOrchestrator(plan)
    results = orchestrator.execute()

    assert len(results) == len(standard_models)
    for res in results:
        assert res.model in standard_models
        assert res.batch_size == 2
        assert res.mean_latency_ms >= 0.0
        assert res.p50_latency_ms >= 0.0
        assert res.p95_latency_ms >= 0.0
        assert res.p99_latency_ms >= 0.0
        # Percentiles monotonic property
        assert res.p95_latency_ms >= res.p50_latency_ms
        assert res.p99_latency_ms >= res.p95_latency_ms
        # Memory metrics
        assert res.peak_memory_mb is not None
        assert res.peak_memory_mb > 0.0
        # Throughput
        assert res.throughput_items_per_sec >= 0.0
