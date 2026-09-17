"""Unit tests for statistical metrics aggregation and structured benchmark reporting."""

import csv
import io
import json

import pytest
import yaml

from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkRunResult
from ml_switcheroo_compiler.benchmarks.metrics import (
    compute_statistical_metrics,
    export_benchmark_results,
)


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
