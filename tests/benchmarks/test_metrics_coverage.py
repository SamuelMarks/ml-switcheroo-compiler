"""Tests for test_metrics_coverage."""

from __future__ import annotations

import csv
import io
import json
import multiprocessing
from typing import Callable, Union

import numpy as np
import pytest
import yaml

import ml_switcheroo_compiler.benchmarks.config_models as cfg_models_mod
import ml_switcheroo_compiler.benchmarks.metrics as metrics_mod
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
    res_single = metrics_mod.compute_statistical_metrics([12.5], batch_size=2, total_flops=1e9, total_bytes_accessed=1e6)
    assert res_single["std_latency_ms"] == 0.0
    assert res_single["mean_latency_ms"] == 12.5
    assert float(res_single["gflops"]) > 0.0
    assert float(res_single["bandwidth_gb_s"]) > 0.0

    # 3. Multiple latencies with percentiles
    lats = [5.0, 10.0, 15.0, 20.0, 25.0]
    res_multi = metrics_mod.compute_statistical_metrics(lats, batch_size=4, peak_memory_mb=64.0, warmup_iters=0)
    assert res_multi["min_latency_ms"] == 5.0
    assert res_multi["max_latency_ms"] == 25.0
    assert res_multi["p50_latency_ms"] == 15.0
    assert float(res_multi["std_latency_ms"]) > 0.0
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
