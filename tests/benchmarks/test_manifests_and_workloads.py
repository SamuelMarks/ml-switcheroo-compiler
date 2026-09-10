"""Tests for benchmark manifests, declarative workloads, and graph construction."""

import os
import tempfile

import pytest

from ml_switcheroo_compiler.benchmarks.config_models import (
    BackendProfile,
    BenchmarkPlan,
    BenchmarkSuite,
    BenchmarkTarget,
    ModelWorkload,
    WorkloadNodeSpec,
    WorkloadTensorSpec,
    build_ir_graph_from_workload,
    load_backend_profiles,
    load_benchmark_plan,
    load_benchmark_suite,
    load_model_workloads,
    load_workloads,
)
from ml_switcheroo_compiler.benchmarks.orchestrator import BenchmarkOrchestrator


def test_load_default_manifests() -> None:
    """Verify loading default model workloads and backend profiles."""
    workloads = load_model_workloads()
    assert "mlp_model" in workloads
    assert "conv_net" in workloads
    assert "attention_block" in workloads

    workloads_alias = load_workloads()
    assert "mlp_model" in workloads_alias

    profiles = load_backend_profiles()
    assert "numpy" in profiles
    assert "pytorch" in profiles
    assert "jax" in profiles
    assert "mlx" in profiles
    assert "cupy" in profiles
    assert isinstance(profiles["numpy"], BackendProfile)


def test_manifest_file_not_found() -> None:
    """Verify FileNotFoundError is raised for non-existent manifest files."""
    with pytest.raises(FileNotFoundError):
        load_model_workloads("/non/existent/path/workloads.yaml")

    with pytest.raises(FileNotFoundError):
        load_backend_profiles("/non/existent/path/profiles.yaml")

    with pytest.raises(FileNotFoundError):
        load_benchmark_suite("/non/existent/path/suite.yaml")

    with pytest.raises(FileNotFoundError):
        load_benchmark_plan("/non/existent/path/plan.yaml")


def test_manifest_invalid_structure() -> None:
    """Verify ValueError is raised when manifest content is not a mapping."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("- item_1\n- item_2\n")
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid model workloads YAML"):
            load_model_workloads(tmp_path)

        with pytest.raises(ValueError, match="Invalid backend profiles YAML"):
            load_backend_profiles(tmp_path)

        with pytest.raises(ValueError, match="Invalid benchmark suite YAML"):
            load_benchmark_suite(tmp_path)

        with pytest.raises(ValueError, match="Invalid benchmark YAML structure"):
            load_benchmark_plan(tmp_path)
    finally:
        os.remove(tmp_path)


def test_load_benchmark_plan_valid() -> None:
    """Verify loading a valid benchmark plan YAML."""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("name: test_plan\nmodels: [mlp_model]\nbatch_sizes: [1]\nwarmup_iterations: 2\nbenchmark_iterations: 5\ntargets: []\n")
        tmp_path = f.name

    try:
        plan = load_benchmark_plan(tmp_path)
        assert plan.name == "test_plan"
        assert plan.warmup_iterations == 2
        assert plan.models == ["mlp_model"]
    finally:
        os.remove(tmp_path)


def test_build_ir_graph_from_workload() -> None:
    """Test constructing LogicalGraph from declarative ModelWorkload."""
    workload = ModelWorkload(
        description="Simple add workload",
        inputs={"in_0": WorkloadTensorSpec(shape=[10, 20], dtype="float32")},
        weights={"w_0": WorkloadTensorSpec(shape=[10, 20], dtype="float32")},
        nodes=[
            WorkloadNodeSpec(
                id="add_0",
                op_type="Add",
                inputs=["in_0", "w_0"],
                attributes={},
            )
        ],
        outputs=["add_0"],
    )

    graph = build_ir_graph_from_workload(workload, name="test_add_graph")
    assert graph.name == "test_add_graph"
    assert "in_0" in graph.nodes
    assert "w_0" in graph.nodes
    assert "add_0" in graph.nodes
    assert graph.nodes["w_0"].attributes.get("is_weight") is True
    assert graph.outputs == ["add_0"]


def test_orchestrator_executes_manifest_workload() -> None:
    """Test BenchmarkOrchestrator running with mlp_model from manifest."""
    plan = BenchmarkPlan(
        name="manifest_mlp_test",
        models=["mlp_model"],
        batch_sizes=[2],
        targets=[BenchmarkTarget(backend="numpy", device="cpu")],
        num_iterations=3,
        warmup_iterations=1,
    )

    orchestrator = BenchmarkOrchestrator(plan)
    results = orchestrator.execute()

    assert len(results) == 1
    res = results[0]
    assert res.model == "mlp_model"
    assert res.batch_size == 2
    assert res.backend == "numpy"
    assert res.mean_latency_ms > 0.0
    assert res.throughput_items_per_sec > 0.0


def test_orchestrator_manifest_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test orchestrator execution when load_model_workloads fails."""
    monkeypatch.setattr(
        "ml_switcheroo_compiler.benchmarks.config_models.load_model_workloads",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Disk failure")),
    )

    plan = BenchmarkPlan(
        name="fallback_test",
        models=["unknown_model"],
        batch_sizes=[1],
        targets=[BenchmarkTarget(backend="numpy", device="cpu")],
        num_iterations=2,
        warmup_iterations=1,
    )

    orchestrator = BenchmarkOrchestrator(plan)
    results = orchestrator.execute()
    assert len(results) == 1
    assert results[0].model == "unknown_model"


def test_load_valid_benchmark_suite() -> None:
    """Test loading a valid BenchmarkSuite from YAML."""
    suite_yaml = """
benchmarks:
  - name: "suite_plan"
    models: ["mlp_model"]
    batch_sizes: [1]
    targets:
      - backend: "numpy"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(suite_yaml)
        tmp_path = f.name

    try:
        suite = load_benchmark_suite(tmp_path)
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.benchmarks) == 1
        assert suite.benchmarks[0].name == "suite_plan"
    finally:
        os.remove(tmp_path)


def test_standard_workloads_manifest() -> None:
    """Verify standard_workloads.yaml exists and contains all required model workloads."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    manifest_path = os.path.join(base_dir, "src", "ml_switcheroo_compiler", "benchmarks", "manifests", "standard_workloads.yaml")
    assert os.path.exists(manifest_path)
    workloads = load_model_workloads(manifest_path)
    expected = [
        "mlp",
        "mlp_model",
        "cnn_resnet_block",
        "conv_net",
        "nanogpt_transformer_block",
        "attention_block",
        "diffusion_resblock",
    ]
    for m in expected:
        assert m in workloads
        assert workloads[m].inputs
        assert workloads[m].nodes
        assert workloads[m].outputs


def test_cross_framework_benchmarking_runs_without_synthetic_fallback_warnings() -> None:
    """Validate cross-framework benchmarking runs across backends without synthetic fallback warnings."""
    import warnings

    plan = BenchmarkPlan(
        name="cross_framework_validation",
        models=["mlp_model"],
        batch_sizes=[1],
        targets=[
            BenchmarkTarget(backend="numpy", device="cpu"),
            BenchmarkTarget(backend="pytorch", device="cpu"),
            BenchmarkTarget(backend="jax", device="cpu"),
            BenchmarkTarget(backend="mlx", device="cpu"),
        ],
        num_iterations=2,
        warmup_iterations=1,
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        orchestrator = BenchmarkOrchestrator(plan)
        results = orchestrator.execute()

    for w in recorded_warnings:
        msg = str(w.message).lower()
        assert "synthetic" not in msg
        assert "fallback" not in msg

    assert len(results) == 4
    backend_names = {r.backend for r in results}
    assert backend_names == {"numpy", "pytorch", "jax", "mlx"}
    for res in results:
        assert res.mean_latency_ms > 0.0
        assert res.p50_latency_ms >= 0.0
        assert res.p95_latency_ms >= 0.0
        assert res.p99_latency_ms >= 0.0
        assert res.throughput_items_per_sec > 0.0
        assert res.peak_memory_mb is not None and res.peak_memory_mb > 0.0


def test_load_model_workloads_fallback_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test load_model_workloads falling back to primary and fallback manifests."""
    real_exists = os.path.exists

    def mock_exists_no_standard(path: str) -> bool:
        if "standard_workloads.yaml" in path:
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", mock_exists_no_standard)
    w_primary = load_model_workloads(None)
    assert "mlp_model" in w_primary

    def mock_exists_fallback(path: str) -> bool:
        if "standard_workloads.yaml" in path or ("workloads.yaml" in path and "model_workloads.yaml" not in path):
            return False
        return real_exists(path)

    monkeypatch.setattr(os.path, "exists", mock_exists_fallback)
    w_fallback = load_model_workloads(None)
    assert "mlp_model" in w_fallback


def test_benchmark_specs_and_constraints() -> None:
    """Test DeviceConstraintSpec, PrecisionTargetSpec, LatencyThresholdSpec instantiation and validation."""
    from ml_switcheroo_compiler.benchmarks.config_models import (
        DeviceConstraintSpec,
        LatencyThresholdSpec,
        PrecisionTargetSpec,
    )

    dc = DeviceConstraintSpec(
        min_memory_mb=1024.0,
        max_memory_mb=8192.0,
        device_types=["cuda", "mps"],
        require_accelerator=True,
    )
    assert dc.min_memory_mb == 1024.0
    assert dc.require_accelerator is True
    assert "mps" in dc.device_types

    pt = PrecisionTargetSpec(
        target_dtype="float16",
        atol=1e-4,
        rtol=1e-3,
    )
    assert pt.target_dtype == "float16"
    assert pt.atol == 1e-4

    lt = LatencyThresholdSpec(
        max_mean_latency_ms=10.0,
        max_p95_latency_ms=15.0,
        max_p99_latency_ms=25.0,
        min_throughput_items_per_sec=100.0,
    )
    assert lt.max_mean_latency_ms == 10.0
    assert lt.min_throughput_items_per_sec == 100.0

    target = BenchmarkTarget(
        backend="numpy",
        device="cpu",
        device_constraints=dc,
        precision_target=pt,
    )
    assert target.device_constraints is not None
    assert target.precision_target is not None

    plan = BenchmarkPlan(
        name="constrained_bench",
        models=["mlp"],
        batch_sizes=[1],
        targets=[target],
        device_constraints=dc,
        precision_target=pt,
        latency_thresholds=lt,
    )
    assert plan.latency_thresholds is not None
    assert plan.latency_thresholds.max_mean_latency_ms == 10.0
