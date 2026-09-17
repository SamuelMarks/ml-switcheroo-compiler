"""Statistical metrics aggregation and structured export for benchmark results."""

from __future__ import annotations

import csv
import io
import json
import math

import yaml

from ml_switcheroo_compiler.benchmarks.config_models import BenchmarkRunResult


def compute_statistical_metrics(
    latencies: list[float],
    batch_size: int = 1,
    peak_memory_mb: float = 0.0,
    warmup_iters: int = 2,
    total_flops: float = 0.0,
    total_bytes_accessed: float = 0.0,
) -> dict[str, list[float] | float]:
    """Calculate comprehensive statistical latency and throughput metrics across iterations.

    Args:
        latencies (list[float]): Raw measured step latencies in milliseconds.
        batch_size (int): Number of items/samples per batch iteration.
        peak_memory_mb (float): Peak memory allocation in megabytes.
        warmup_iters (int): Warmup iteration count.
        total_flops (float): Total floating point operations performed per step.
        total_bytes_accessed (float): Memory buffer bytes accessed per step.

    Returns:
        dict[str, Union[list[float], float]]: Comprehensive metrics payload.
    """
    sorted_lat: list[float] = sorted(latencies) if latencies else [0.0]
    n: int = len(latencies)
    mean_lat: float = float(sum(latencies) / n) if n > 0 else 0.0

    p50_lat: float = float(sorted_lat[int(len(sorted_lat) * 0.50)])
    p90_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.90))])
    p95_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.95))])
    p99_lat: float = float(sorted_lat[min(len(sorted_lat) - 1, int(len(sorted_lat) * 0.99))])
    min_lat: float = float(sorted_lat[0])
    max_lat: float = float(sorted_lat[-1])

    # Sample standard deviation
    if n > 1:
        variance = sum((x - mean_lat) ** 2 for x in latencies) / (n - 1)
        std_lat = float(math.sqrt(variance))
    else:
        std_lat = 0.0

    mean_sec: float = mean_lat / 1000.0 if mean_lat > 0 else 1e-9
    throughput: float = float(batch_size / mean_sec)
    gflops: float = float((total_flops / mean_sec) / 1e9) if total_flops > 0 else 0.0
    bandwidth_gb_s: float = float((total_bytes_accessed / mean_sec) / 1e9) if total_bytes_accessed > 0 else 0.0

    return {
        "latencies": latencies,
        "latency_ms": mean_lat,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50_lat,
        "p90_latency_ms": p90_lat,
        "p95_latency_ms": p95_lat,
        "p99_latency_ms": p99_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "std_latency_ms": std_lat,
        "throughput_items_per_sec": throughput,
        "gflops": gflops,
        "bandwidth_gb_s": bandwidth_gb_s,
        "peak_memory_mb": peak_memory_mb,
        "warmup_iterations": max(1, warmup_iters),
    }


def export_benchmark_results(
    results: list[BenchmarkRunResult],
    format: str = "json",
) -> str:
    """Export benchmark run results to structured text format (JSON, YAML, CSV).

    Args:
        results (list[BenchmarkRunResult]): Collection of benchmark run results.
        format (str): Target format: 'json', 'yaml', or 'csv'. Defaults to 'json'.

    Returns:
        str: Serialized benchmark report string.

    Raises:
        ValueError: If format is unsupported.
    """
    normalized_format: str = format.strip().lower()

    if normalized_format == "json":
        data = [r.model_dump() for r in results]
        return json.dumps(data, indent=2)

    if normalized_format == "yaml":
        data = [r.model_dump() for r in results]
        return str(yaml.safe_dump(data, sort_keys=False))

    if normalized_format == "csv":
        output = io.StringIO()
        fieldnames = [
            "model",
            "batch_size",
            "backend",
            "device",
            "mean_latency_ms",
            "p50_latency_ms",
            "p90_latency_ms",
            "p95_latency_ms",
            "p99_latency_ms",
            "min_latency_ms",
            "max_latency_ms",
            "std_latency_ms",
            "throughput_items_per_sec",
            "gflops",
            "bandwidth_gb_s",
            "peak_memory_mb",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for res in results:
            writer.writerow(res.model_dump())
        return output.getvalue()

    raise ValueError(f"Unsupported export format '{format}'. Supported formats: 'json', 'yaml', 'csv'.")
