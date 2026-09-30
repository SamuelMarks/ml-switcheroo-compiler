"""Tests for test_config_models_coverage."""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from typing import Callable, Union

import numpy as np
import pytest

import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
from ml_switcheroo_compiler.ir.core import IRGraph

MetricsVal = Union[list[float], float]

ProcessArg = Union[tuple[str, str], IRGraph, dict[str, np.ndarray], str, int, multiprocessing.Queue, None]


class MockSimpleProfiler:
    """Mock profiler providing profile_graph implementation."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray]): Inputs.
            device (str | None): Device identifier.
            num_iters (int): Measurement count.
            warmup_iters (int): Warmup count.

        Returns:
            dict[str, MetricsVal]: Profiler metrics.
        """
        return {"latencies": [1.0, 2.0], "peak_memory_mb": 10.0}


class MockBackendClass:
    """Mock backend implementation."""

    def compile(self) -> None:
        """Compile backend."""
        pass


class MockDeterministicQueue:
    """Deterministic queue mock avoiding multiprocessing pipe latency."""

    def __init__(self, items: list[tuple[str, Union[dict[str, MetricsVal], str]]]) -> None:
        """Initialize mock queue with payload items.

        Args:
            items (list[tuple[str, Union[dict[str, MetricsVal], str]]]): Items to queue.
        """
        self.items: list[tuple[str, Union[dict[str, MetricsVal], str]]] = list(items)

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            bool: True if empty.
        """
        return len(self.items) == 0

    def get(self) -> tuple[str, Union[dict[str, MetricsVal], str]]:
        """Retrieve next item.

        Returns:
            tuple[str, Union[dict[str, MetricsVal], str]]: Item.
        """
        return self.items.pop(0)

    def put(self, item: tuple[str, Union[dict[str, MetricsVal], str]]) -> None:
        """Put item into queue.

        Args:
            item (tuple[str, Union[dict[str, MetricsVal], str]]): Item.
        """
        self.items.append(item)


class MockProcessOk:
    """Mock multiprocessing process simulating clean execution."""

    def __init__(self, target: Callable[..., None], args: tuple[ProcessArg, ...]) -> None:
        """Initialize mock process.

        Args:
            target (Callable[..., None]): Target function.
            args (tuple[ProcessArg, ...]): Process arguments.
        """
        self._target: Callable[..., None] = target
        self._args: tuple[ProcessArg, ...] = args

    def start(self) -> None:
        """Start process."""
        pass

    def join(self, timeout: int | None = None) -> None:
        """Wait for process completion.

        Args:
            timeout (int | None): Timeout duration in seconds.
        """
        pass

    def is_alive(self) -> bool:
        """Check if process is alive.

        Returns:
            bool: False for completed process.
        """
        return False

    def terminate(self) -> None:
        """Terminate process."""
        pass


class MockProcessTimeout(MockProcessOk):
    """Mock multiprocessing process simulating timeout."""

    def is_alive(self) -> bool:
        """Check if process is alive.

        Returns:
            bool: Always True to simulate timeout.
        """
        return True


def _raise_load_err(p: str = "") -> dict[str, cfg_models_mod.BackendProfile]:
    """Helper to simulate load_backend_profiles error.

    Args:
        p (str): Unused path.

    Returns:
        dict[str, cfg_models_mod.BackendProfile]: Never returned.

    Raises:
        ValueError: Always raised.
    """
    raise ValueError("boom")


def _raise_workload_err() -> dict[str, cfg_models_mod.ModelWorkload]:
    """Helper to simulate load_model_workloads error.

    Returns:
        dict[str, cfg_models_mod.ModelWorkload]: Never returned.

    Raises:
        ValueError: Always raised.
    """
    raise ValueError("cannot load workloads")


class MockDeviceFailingProfiler:
    """Mock profiler that rejects non-None device kwarg and accepts device=None."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray],
        *args: int,
        **kwargs: str | int | None,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution with TypeError when device is provided.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray]): Inputs.
            *args (int): Additional positional args.
            **kwargs (str | int | None): Keyword arguments.

        Returns:
            dict[str, MetricsVal]: Profiler metrics.

        Raises:
            TypeError: When device is not None.
        """
        if kwargs.get("device") is not None:
            raise TypeError("Device argument not accepted")
        return {"latencies": [0.0], "peak_memory_mb": 0.0}


def test_config_models_to_json_and_yaml() -> None:
    """Verify BenchmarkRunResult to_json and to_yaml methods."""
    item = cfg_models_mod.BenchmarkRunResult(
        model="cnn",
        batch_size=16,
        backend="numpy",
        device="cpu",
        mean_latency_ms=5.0,
        p50_latency_ms=5.0,
        p95_latency_ms=5.5,
        p99_latency_ms=6.0,
        peak_memory_mb=256.0,
        throughput_items_per_sec=3200.0,
    )
    js = item.to_json()
    assert '"cnn"' in js
    ym = item.to_yaml()
    assert "model: cnn" in ym


def test_config_models_loaders() -> None:
    """Verify YAML configuration loaders and error handling for plans and manifests."""
    # 1. load_perf_baselines with default and custom path
    baselines_default = cfg_models_mod.load_perf_baselines()
    assert isinstance(baselines_default, cfg_models_mod.PerfBaselinesManifestModel)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(
            """baselines:
  test_model:
    workload_name: test
    backend: numpy
    baseline_mean_latency_ms: 1.0
"""
        )
        temp_name = tf.name

    try:
        baselines_custom = cfg_models_mod.load_perf_baselines(temp_name)
        assert "test_model" in baselines_custom.baselines
    finally:
        os.unlink(temp_name)

    # 2. load_benchmark_plan
    with pytest.raises(FileNotFoundError):
        cfg_models_mod.load_benchmark_plan("nonexistent_plan.yaml")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""- not a dict
""")
        invalid_yaml = tf.name

    try:
        with pytest.raises(ValueError, match="Invalid benchmark YAML structure"):
            cfg_models_mod.load_benchmark_plan(invalid_yaml)
    finally:
        os.unlink(invalid_yaml)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""name: valid_plan
models: [m1]
batch_sizes: [1]
targets:
  - backend: numpy
""")
        valid_plan_yaml = tf.name

    try:
        plan_loaded = cfg_models_mod.load_benchmark_plan(valid_plan_yaml)
        assert plan_loaded.name == "valid_plan"
    finally:
        os.unlink(valid_plan_yaml)

    # 3. load_benchmark_suite
    with pytest.raises(FileNotFoundError):
        cfg_models_mod.load_benchmark_suite("nonexistent_suite.yaml")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""- not a dict
""")
        invalid_suite = tf.name

    try:
        with pytest.raises(ValueError, match="Invalid benchmark suite YAML structure"):
            cfg_models_mod.load_benchmark_suite(invalid_suite)
    finally:
        os.unlink(invalid_suite)

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""benchmarks:
  - name: plan_in_suite
    models: [m]
    batch_sizes: [1]
    targets:
      - backend: numpy
""")
        valid_suite_yaml = tf.name

    try:
        suite_loaded = cfg_models_mod.load_benchmark_suite(valid_suite_yaml)
        assert len(suite_loaded.benchmarks) == 1
    finally:
        os.unlink(valid_suite_yaml)

    # 4. load_model_workloads
    workloads = cfg_models_mod.load_workloads()
    assert len(workloads) > 0

    with pytest.raises(FileNotFoundError):
        cfg_models_mod.load_model_workloads("nonexistent_workloads.yaml")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""- invalid list
""")
        invalid_workload = tf.name

    try:
        with pytest.raises(ValueError, match="Invalid model workloads YAML"):
            cfg_models_mod.load_model_workloads(invalid_workload)
    finally:
        os.unlink(invalid_workload)

    # Primary, fallback, and missing branches in load_model_workloads
    with pytest.MonkeyPatch.context() as mp:
        orig_exists = os.path.exists

        def mock_exists_primary(path: str) -> bool:
            if path.endswith("/standard_workloads.yaml"):
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_primary)
        workloads_primary = cfg_models_mod.load_model_workloads()
        assert len(workloads_primary) > 0

        def mock_exists_fallback(path: str) -> bool:
            if path.endswith("/standard_workloads.yaml") or path.endswith("/workloads.yaml"):
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_fallback)
        workloads_fallback = cfg_models_mod.load_model_workloads()
        assert len(workloads_fallback) > 0

        def mock_exists_none(path: str) -> bool:
            if any(path.endswith("/" + name) for name in ("standard_workloads.yaml", "workloads.yaml", "model_workloads.yaml")):
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_none)
        with pytest.raises(FileNotFoundError):
            cfg_models_mod.load_model_workloads()

    # 5. load_backend_profiles_manifest and load_backend_profiles
    profiles = cfg_models_mod.load_backend_profiles()
    assert "numpy" in profiles

    with pytest.raises(FileNotFoundError):
        cfg_models_mod.load_backend_profiles_manifest("nonexistent_profiles.yaml")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""- invalid
""")
        invalid_prof = tf.name

    try:
        with pytest.raises(ValueError, match="Invalid backend profiles YAML"):
            cfg_models_mod.load_backend_profiles_manifest(invalid_prof)
    finally:
        os.unlink(invalid_prof)

    # load_backend_profiles with custom path and default manifest missing branch
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write("""profiles:
  custom_numpy:
    timer: perf_counter
    sync_method: none
    default_warmup_iterations: 1
    default_measured_iterations: 1
    memory_tracking: none
    profiler_module: ml_switcheroo_compiler.backends.numpy.profiler
    profiler_class: NumpyProfiler
""")
        custom_prof_yaml = tf.name

    try:
        custom_profs = cfg_models_mod.load_backend_profiles(custom_prof_yaml)
        assert "custom_numpy" in custom_profs
    finally:
        os.unlink(custom_prof_yaml)

    with pytest.MonkeyPatch.context() as mp:
        orig_exists = os.path.exists

        def mock_exists_no_bp(path: str) -> bool:
            if path.endswith("/backend_profiles.yaml"):
                return False
            return orig_exists(path)

        mp.setattr(os.path, "exists", mock_exists_no_bp)
        with pytest.raises(FileNotFoundError):
            cfg_models_mod.load_backend_profiles_manifest(None)
