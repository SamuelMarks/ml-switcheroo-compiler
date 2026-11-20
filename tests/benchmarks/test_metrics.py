"""Unit tests for statistical metrics aggregation and structured benchmark reporting."""

from __future__ import annotations

import csv
import io
import json
import typing

import numpy as np
import pytest
import yaml
from ml_switcheroo_ir import IRGraph

import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
import ml_switcheroo_compiler.benchmarks.metrics as metrics_mod
from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkRunResult
from ml_switcheroo_compiler.benchmarks.metrics import compute_statistical_metrics, export_benchmark_results
from ml_switcheroo_compiler.core.shape import TypedVariadicCallable
from tests.testing_utils import coerce_float


def test_compute_statistical_metrics_standard() -> None:
    """Verify statistical metrics calculation across standard latencies."""
    latencies = [10.0, 12.0, 11.0, 14.0, 13.0]
    metrics = compute_statistical_metrics(
        latencies=latencies,
        batch_size=8,
        peak_memory_mb=64.5,
        warmup_iters=3,
        total_flops=1e9,
        total_bytes_accessed=1e8,
    )
    assert metrics["latency_ms"] == 12.0
    assert metrics["mean_latency_ms"] == 12.0
    assert metrics["p50_latency_ms"] == 12.0
    assert metrics["min_latency_ms"] == 10.0
    assert metrics["max_latency_ms"] == 14.0
    assert metrics["peak_memory_mb"] == 64.5
    assert metrics["warmup_iterations"] == 3
    assert metrics["throughput_items_per_sec"] > 0.0
    assert metrics["gflops"] > 0.0
    assert metrics["bandwidth_gb_s"] > 0.0
    assert metrics["std_latency_ms"] > 0.0


def test_compute_statistical_metrics_empty_and_single() -> None:
    """Verify edge cases with empty latency list or single measurement."""
    empty_metrics = compute_statistical_metrics([])
    assert empty_metrics["mean_latency_ms"] == 0.0
    assert empty_metrics["std_latency_ms"] == 0.0

    single_metrics = compute_statistical_metrics([5.0])
    assert single_metrics["mean_latency_ms"] == 5.0
    assert single_metrics["p50_latency_ms"] == 5.0
    assert single_metrics["std_latency_ms"] == 0.0


def test_benchmark_run_result_serialization() -> None:
    """Verify BenchmarkRunResult JSON and YAML serialization methods."""
    result = BenchmarkRunResult(
        model="mlp",
        batch_size=32,
        backend="numpy",
        device="cpu",
        mean_latency_ms=2.5,
        p50_latency_ms=2.4,
        p90_latency_ms=2.8,
        p95_latency_ms=3.0,
        p99_latency_ms=3.2,
        peak_memory_mb=128.0,
        throughput_items_per_sec=12800.0,
        gflops=50.0,
    )

    json_str = result.to_json()
    parsed_json = json.loads(json_str)
    assert parsed_json["model"] == "mlp"
    assert parsed_json["batch_size"] == 32
    assert parsed_json["gflops"] == 50.0

    yaml_str = result.to_yaml()
    parsed_yaml = yaml.safe_load(yaml_str)
    assert parsed_yaml["model"] == "mlp"
    assert parsed_yaml["mean_latency_ms"] == 2.5


def test_export_benchmark_results_formats() -> None:
    """Verify export_benchmark_results supports JSON, YAML, and CSV outputs."""
    res1 = BenchmarkRunResult(
        model="mlp",
        batch_size=1,
        backend="pytorch",
        device="cpu",
        mean_latency_ms=1.5,
        p50_latency_ms=1.4,
        p95_latency_ms=1.7,
        p99_latency_ms=1.9,
        peak_memory_mb=64.0,
        throughput_items_per_sec=666.6,
    )
    res2 = BenchmarkRunResult(
        model="convnet",
        batch_size=4,
        backend="jax",
        device="cpu",
        mean_latency_ms=4.0,
        p50_latency_ms=3.9,
        p95_latency_ms=4.2,
        p99_latency_ms=4.5,
        peak_memory_mb=96.0,
        throughput_items_per_sec=1000.0,
    )
    results = [res1, res2]

    # JSON export
    json_out = export_benchmark_results(results, format="json")
    parsed_json = json.loads(json_out)
    assert len(parsed_json) == 2
    assert parsed_json[0]["model"] == "mlp"
    assert parsed_json[1]["model"] == "convnet"

    # YAML export
    yaml_out = export_benchmark_results(results, format="yaml")
    parsed_yaml = yaml.safe_load(yaml_out)
    assert len(parsed_yaml) == 2
    assert parsed_yaml[0]["backend"] == "pytorch"

    # CSV export
    csv_out = export_benchmark_results(results, format="csv")
    reader = csv.DictReader(io.StringIO(csv_out))
    rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["model"] == "mlp"
    assert rows[1]["backend"] == "jax"
    assert "mean_latency_ms" in rows[0]

    # Unsupported format
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_benchmark_results(results, format="parquet")


MetricsVal = typing.Union[list[float], "float"]

ProcessArg = typing.Union[
    tuple[str, str], IRGraph, dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]], str, int, BackendArg, "None"
]


class MockSimpleProfiler:
    """Mock profiler providing profile_graph implementation."""

    def profile_graph(
        self,
        graph: IRGraph,
        inputs: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
        device: str | None = None,
        num_iters: int = 10,
        warmup_iters: int = 2,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]): Inputs.
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

    def __init__(self, items: list[tuple[str, dict[str, MetricsVal] | str]]) -> None:
        """Initialize mock queue with payload items.

        Args:
            items (list[tuple[str, Union[dict[str, MetricsVal], str]]]): Items to queue.
        """
        self.items: list[tuple[str, dict[str, MetricsVal] | str]] = list(items)

    def empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            bool: True if empty.
        """
        return len(self.items) == 0

    def get(self) -> tuple[str, dict[str, MetricsVal] | str]:
        """Retrieve next item.

        Returns:
            tuple[str, Union[dict[str, MetricsVal], str]]: Item.
        """
        return self.items.pop(0)

    def put(self, item: tuple[str, dict[str, MetricsVal] | str]) -> None:
        """Put item into queue.

        Args:
            item (tuple[str, Union[dict[str, MetricsVal], str]]): Item.
        """
        self.items.append(item)


class MockProcessOk:
    """Mock multiprocessing process simulating clean execution."""

    def __init__(self, target: TypedVariadicCallable[None], args: tuple[ProcessArg, ...]) -> None:
        """Initialize mock process.

        Args:
            target (TypedVariadicCallable[None]): Target function.
            args (tuple[ProcessArg, ...]): Process arguments.
        """
        self._target: TypedVariadicCallable[None] = target
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
        inputs: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]],
        *args: int,
        **kwargs: str | int | None,
    ) -> dict[str, MetricsVal]:
        """Profile graph execution with TypeError when device is provided.

        Args:
            graph (IRGraph): Computational graph.
            inputs (dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]): Inputs.
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


def test_metrics_empty_and_single() -> None:
    """Verify metrics calculation on empty, single, and multiple iteration lists."""
    # 1. Empty latencies
    res_empty = metrics_mod.compute_statistical_metrics([], batch_size=1, peak_memory_mb=0.0)
    assert res_empty["latencies"] == []
    assert res_empty["mean_latency_ms"] == 0.0
    assert res_empty["std_latency_ms"] == 0.0
    assert res_empty["gflops"] == 0.0
    assert res_empty["bandwidth_gb_s"] == 0.0

    # 2. Single latency
    res_single = metrics_mod.compute_statistical_metrics(
        [12.5], batch_size=2, total_flops=1e9, total_bytes_accessed=1e6
    )
    assert res_single["std_latency_ms"] == 0.0
    assert res_single["mean_latency_ms"] == 12.5
    assert coerce_float(res_single["gflops"]) > 0.0
    assert coerce_float(res_single["bandwidth_gb_s"]) > 0.0

    # 3. Multiple latencies with percentiles
    lats = [5.0, 10.0, 15.0, 20.0, 25.0]
    res_multi = metrics_mod.compute_statistical_metrics(lats, batch_size=4, peak_memory_mb=64.0, warmup_iters=0)
    assert res_multi["min_latency_ms"] == 5.0
    assert res_multi["max_latency_ms"] == 25.0
    assert res_multi["p50_latency_ms"] == 15.0
    assert coerce_float(res_multi["std_latency_ms"]) > 0.0
    assert res_multi["warmup_iterations"] == 1


def test_metrics_export_formats() -> None:
    """Verify JSON, YAML, and CSV export and unsupported format rejection."""
    item = cfg_models_mod.BenchmarkRunResult(
        model="mlp",
        batch_size=8,
        backend="numpy",
        device="cpu",
        mean_latency_ms=10.0,
        p50_latency_ms=9.5,
        p95_latency_ms=11.0,
        p99_latency_ms=12.0,
        peak_memory_mb=128.0,
        throughput_items_per_sec=800.0,
    )
    items = [item]

    # JSON export
    json_out = metrics_mod.export_benchmark_results(items, format="json")
    parsed_json = json.loads(json_out)
    assert len(parsed_json) == 1
    assert parsed_json[0]["model"] == "mlp"

    # YAML export
    yaml_out = metrics_mod.export_benchmark_results(items, format="yaml")
    parsed_yaml = yaml.safe_load(yaml_out)
    assert len(parsed_yaml) == 1
    assert parsed_yaml[0]["backend"] == "numpy"

    # CSV export
    csv_out = metrics_mod.export_benchmark_results(items, format="csv")
    reader = list(csv.DictReader(io.StringIO(csv_out)))
    assert len(reader) == 1
    assert reader[0]["model"] == "mlp"

    # Unsupported format
    with pytest.raises(ValueError, match="Unsupported export format"):
        metrics_mod.export_benchmark_results(items, format="xml")
