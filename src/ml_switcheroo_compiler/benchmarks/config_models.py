"""Data models for benchmark plans, configurations, and performance results."""

import os
from typing import Optional, Union

import yaml
from ml_switcheroo_ir import LogicalGraph, LogicalNode
from pydantic import BaseModel, Field

CompileOptionValue = Union[str, int, float, bool]
NodeAttributeValue = Union[str, int, float, bool, list[int], list[float]]


class DeviceConstraintSpec(BaseModel):
    """Specification of device constraints for execution profiling.

    Attributes:
        min_memory_mb (Optional[float]): Minimum memory required in megabytes.
        max_memory_mb (Optional[float]): Maximum memory ceiling permitted in megabytes.
        device_types (list[str]): Permitted device types (e.g. ['cpu', 'cuda', 'mps']).
        require_accelerator (bool): Whether execution requires an accelerator.
    """

    min_memory_mb: Optional[float] = None
    max_memory_mb: Optional[float] = None
    device_types: list[str] = Field(default_factory=lambda: ["cpu", "cuda", "mps"])
    require_accelerator: bool = False


DeviceConstraints = DeviceConstraintSpec


class PrecisionTargetSpec(BaseModel):
    """Specification of precision targets for benchmarking numerical fidelity.

    Attributes:
        target_dtype (str): Target floating point precision string (e.g. 'float32', 'float16').
        atol (float): Absolute tolerance for numerical parity verification.
        rtol (float): Relative tolerance for numerical parity verification.
    """

    target_dtype: str = "float32"
    atol: float = 1e-5
    rtol: float = 1e-5


PrecisionTarget = PrecisionTargetSpec


class LatencyThresholdSpec(BaseModel):
    """Specification of latency threshold constraints for execution benchmarks.

    Attributes:
        max_mean_latency_ms (Optional[float]): Maximum allowed mean latency in milliseconds.
        max_p95_latency_ms (Optional[float]): Maximum allowed 95th percentile latency in milliseconds.
        max_p99_latency_ms (Optional[float]): Maximum allowed 99th percentile latency in milliseconds.
        min_throughput_items_per_sec (Optional[float]): Minimum throughput in items per second.
    """

    max_mean_latency_ms: Optional[float] = None
    max_p95_latency_ms: Optional[float] = None
    max_p99_latency_ms: Optional[float] = None
    min_throughput_items_per_sec: Optional[float] = None


LatencyThreshold = LatencyThresholdSpec


class BenchmarkTarget(BaseModel):
    """Configuration for a benchmark target execution environment.

    Attributes:
        backend (str): Target execution backend name (e.g. 'numpy', 'pytorch').
        device (Optional[str]): Device identifier (e.g. 'cpu', 'cuda', 'mps').
        compile_options (dict[str, Union[str, int, float, bool]]): Extra compilation options.
        device_constraints (Optional[DeviceConstraintSpec]): Optional device constraint bounds.
        precision_target (Optional[PrecisionTargetSpec]): Target numerical precision requirements.
    """

    backend: str
    device: Optional[str] = None
    compile_options: dict[str, CompileOptionValue] = Field(default_factory=dict)
    device_constraints: Optional[DeviceConstraintSpec] = None
    precision_target: Optional[PrecisionTargetSpec] = None


class BenchmarkPlan(BaseModel):
    """Declarative plan for running a cross-backend benchmark.

    Attributes:
        name (str): Identifier name for the benchmark scenario.
        description (Optional[str]): Human-readable description of the benchmark.
        models (list[str]): List of model or graph identifiers to execute.
        batch_sizes (list[int]): List of batch sizes to sweep over.
        targets (list[BenchmarkTarget]): List of backend/device targets to compare.
        num_iterations (int): Number of measurement iterations per batch size.
        warmup_iterations (int): Number of warmup iterations before measuring.
        input_shapes (dict[str, list[int]]): Default tensor dimensions per input.
        dtypes (list[str]): Data types to sweep over.
        device_constraints (Optional[DeviceConstraintSpec]): Plan-level device constraints.
        precision_target (Optional[PrecisionTargetSpec]): Plan-level precision target requirements.
        latency_thresholds (Optional[LatencyThresholdSpec]): Plan-level latency performance thresholds.
        isolate_process (bool): Whether to isolate execution out-of-process.
    """

    name: str
    description: Optional[str] = None
    models: list[str]
    batch_sizes: list[int]
    targets: list[BenchmarkTarget]
    num_iterations: int = 100
    warmup_iterations: int = 10
    input_shapes: dict[str, list[int]] = Field(default_factory=dict)
    dtypes: list[str] = Field(default_factory=lambda: ["float32"])
    device_constraints: Optional[DeviceConstraintSpec] = None
    precision_target: Optional[PrecisionTargetSpec] = None
    latency_thresholds: Optional[LatencyThresholdSpec] = None
    isolate_process: bool = False


class BenchmarkSuite(BaseModel):
    """Suite containing multiple declarative benchmark plans.

    Attributes:
        benchmarks (list[BenchmarkPlan]): Collection of benchmark plans.
    """

    benchmarks: list[BenchmarkPlan] = Field(default_factory=list)


class BenchmarkRunResult(BaseModel):
    """Result of a single benchmark run capturing latency and memory metrics.

    Attributes:
        model (str): Name of the executed model.
        batch_size (int): Batch size used during the run.
        backend (str): Target backend that executed the model.
        device (str): Device target utilized.
        mean_latency_ms (float): Mean execution latency in milliseconds.
        p50_latency_ms (float): 50th percentile latency in milliseconds.
        p95_latency_ms (float): 95th percentile latency in milliseconds.
        p99_latency_ms (float): 99th percentile latency in milliseconds.
        peak_memory_mb (Optional[float]): Peak memory allocation in megabytes.
        throughput_items_per_sec (float): Processed items per second.
    """

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


class WorkloadTensorSpec(BaseModel):
    """Specification of a tensor in a benchmark workload.

    Attributes:
        shape (list[int]): Dimensions of the tensor.
        dtype (str): Data type string representation (e.g. 'float32').
    """

    shape: list[int]
    dtype: str = "float32"


class WorkloadNodeSpec(BaseModel):
    """Specification of a single operation node in a benchmark workload graph.

    Attributes:
        id (str): Unique node identifier.
        op_type (str): Operation name (e.g. 'MatMul', 'Relu').
        inputs (list[str]): Node IDs providing inputs to this operation.
        attributes (dict[str, Union[str, int, float, bool, list[int], list[float]]]): Operation attributes.
    """

    id: str
    op_type: str
    inputs: list[str] = Field(default_factory=list)
    attributes: dict[str, NodeAttributeValue] = Field(default_factory=dict)


class ModelWorkload(BaseModel):
    """Declarative specification for a model workload graph.

    Attributes:
        description (Optional[str]): Description of the model workload.
        batch_sizes (list[int]): Recommended batch sizes for benchmark sweeps.
        inputs (dict[str, WorkloadTensorSpec]): Named model input tensor specifications.
        weights (dict[str, WorkloadTensorSpec]): Static weight tensor specifications.
        nodes (list[WorkloadNodeSpec]): Sequential/DAG operations defining the model.
        outputs (list[str]): Output node IDs.
    """

    description: Optional[str] = None
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32])
    inputs: dict[str, WorkloadTensorSpec]
    weights: dict[str, WorkloadTensorSpec] = Field(default_factory=dict)
    nodes: list[WorkloadNodeSpec]
    outputs: list[str]


class ModelWorkloadManifest(BaseModel):
    """Manifest containing multiple model workloads.

    Attributes:
        models (dict[str, ModelWorkload]): Map of workload identifiers to model specifications.
    """

    models: dict[str, ModelWorkload] = Field(default_factory=dict)


class BackendProfile(BaseModel):
    """Hardware and runtime profiling configuration for a specific backend.

    Attributes:
        timer (str): High-resolution timer mechanism to use.
        sync_method (str): Hardware synchronization primitive.
        default_warmup_iterations (int): Recommended warmup iterations.
        default_measured_iterations (int): Recommended measured iterations.
        memory_tracking (str): Strategy for reporting memory utilization.
    """

    timer: str
    sync_method: str
    default_warmup_iterations: int
    default_measured_iterations: int
    memory_tracking: str


class BackendProfilesManifest(BaseModel):
    """Manifest containing execution profiles for each supported backend.

    Attributes:
        profiles (dict[str, BackendProfile]): Map of backend name to profile config.
    """

    profiles: dict[str, BackendProfile] = Field(default_factory=dict)


def load_benchmark_plan(yaml_path: str) -> BenchmarkPlan:
    """Load a single BenchmarkPlan from a YAML file path.

    Args:
        yaml_path (str): Filesystem path to the YAML file.

    Returns:
        BenchmarkPlan: Validated benchmark plan model.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If YAML parsing produces invalid data.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Benchmark YAML not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark YAML structure in {yaml_path}")
    return BenchmarkPlan(**data)


def load_benchmark_suite(yaml_path: str) -> BenchmarkSuite:
    """Load a BenchmarkSuite containing multiple plans from a YAML file.

    Args:
        yaml_path (str): Filesystem path to the YAML file.

    Returns:
        BenchmarkSuite: Validated benchmark suite model.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If YAML parsing produces invalid data.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Benchmark YAML not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid benchmark suite YAML structure in {yaml_path}")
    return BenchmarkSuite(**data)


def load_model_workloads(yaml_path: Optional[str] = None) -> dict[str, ModelWorkload]:
    """Load model workloads from declarative YAML manifest.

    Args:
        yaml_path (Optional[str]): Path to model workloads YAML, defaults to internal manifest.

    Returns:
        dict[str, ModelWorkload]: Map of model identifiers to workload definitions.

    Raises:
        FileNotFoundError: If the manifest path does not exist.
        ValueError: If YAML contents are invalid.
    """
    if yaml_path is None:
        base_dir = os.path.dirname(__file__)
        standard_path = os.path.join(base_dir, "manifests", "standard_workloads.yaml")
        primary_path = os.path.join(base_dir, "manifests", "workloads.yaml")
        fallback_path = os.path.join(base_dir, "manifests", "model_workloads.yaml")
        if os.path.exists(standard_path):
            yaml_path = standard_path
        elif os.path.exists(primary_path):
            yaml_path = primary_path
        else:
            yaml_path = fallback_path
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Model workloads YAML not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid model workloads YAML in {yaml_path}")
    manifest = ModelWorkloadManifest(**data)
    return manifest.models


def load_workloads(yaml_path: Optional[str] = None) -> dict[str, ModelWorkload]:
    """Load workloads from declarative workloads manifest.

    Args:
        yaml_path (Optional[str]): Path to workloads YAML, defaults to internal manifest.

    Returns:
        dict[str, ModelWorkload]: Map of model identifiers to workload definitions.
    """
    return load_model_workloads(yaml_path)


def load_backend_profiles(yaml_path: Optional[str] = None) -> dict[str, BackendProfile]:
    """Load backend profiles from declarative YAML manifest.

    Args:
        yaml_path (Optional[str]): Path to backend profiles YAML, defaults to internal manifest.

    Returns:
        dict[str, BackendProfile]: Map of backend names to execution profiles.

    Raises:
        FileNotFoundError: If the manifest path does not exist.
        ValueError: If YAML contents are invalid.
    """
    if yaml_path is None:
        base_dir = os.path.dirname(__file__)
        yaml_path = os.path.join(base_dir, "manifests", "backend_profiles.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Backend profiles YAML not found: {yaml_path}")
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid backend profiles YAML in {yaml_path}")
    manifest = BackendProfilesManifest(**data)
    return manifest.profiles


def build_ir_graph_from_workload(workload: ModelWorkload, name: str = "workload_graph") -> LogicalGraph:
    """Construct an executable LogicalGraph from a declarative ModelWorkload definition.

    Args:
        workload (ModelWorkload): Declarative model workload specification.
        name (str): Name for the generated graph.

    Returns:
        LogicalGraph: Constructed and populated IR graph ready for compilation or evaluation.
    """
    graph = LogicalGraph(name=name)
    graph.inputs = []

    # Register input nodes
    for inp_id, inp_spec in workload.inputs.items():
        node = LogicalNode(
            id=inp_id,
            op_type="Input",
            inputs=[],
            attributes={"shape": inp_spec.shape, "dtype": inp_spec.dtype},
            shape_metadata=tuple(inp_spec.shape),
        )
        graph.nodes[inp_id] = node
        graph.inputs.append(inp_id)

    # Register weight nodes as Constant nodes
    for w_id, w_spec in workload.weights.items():
        node = LogicalNode(
            id=w_id,
            op_type="Input",
            inputs=[],
            attributes={"shape": w_spec.shape, "dtype": w_spec.dtype, "is_weight": True},
            shape_metadata=tuple(w_spec.shape),
        )
        graph.nodes[w_id] = node
        graph.inputs.append(w_id)

    # Register operation nodes
    for node_spec in workload.nodes:
        attrs: dict[str, NodeAttributeValue] = dict(node_spec.attributes)
        node = LogicalNode(
            id=node_spec.id,
            op_type=node_spec.op_type,
            inputs=list(node_spec.inputs),
            attributes=attrs,
        )
        graph.nodes[node_spec.id] = node

    graph.outputs = list(workload.outputs)
    return graph
