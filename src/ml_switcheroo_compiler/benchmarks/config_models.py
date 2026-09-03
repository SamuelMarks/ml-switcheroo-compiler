"""Data models for benchmark plans and results."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class BenchmarkTarget(BaseModel):
    """Configuration for a benchmark target."""

    backend: str
    device: Optional[str] = None
    compile_options: dict[str, Any] = Field(default_factory=dict)


class BenchmarkPlan(BaseModel):
    """Declarative plan for running a cross-backend benchmark."""

    name: str
    description: Optional[str] = None
    models: list[str]  # e.g., references to graph builder functions or serialized graphs
    batch_sizes: list[int]
    targets: list[BenchmarkTarget]
    num_iterations: int = 100
    warmup_iterations: int = 10


class BenchmarkRunResult(BaseModel):
    """Result of a single benchmark run."""

    model: str
    batch_size: int
    backend: str
    device: str
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    peak_memory_mb: Optional[float] = None
    throughput_items_per_sec: float
