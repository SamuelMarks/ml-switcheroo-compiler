"""Pydantic models for configuration files."""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class NodePatternConfig(BaseModel):
    """Configuration for an IR node pattern."""

    op_type: Optional[str] = None
    capture: Optional[str] = None
    inputs: Optional[list["NodePatternConfig"]] = None
    wildcard: bool = False
    commutative: bool = False
    broadcast_dims: bool = False


class ReplacementConfig(BaseModel):
    """Configuration for replacement of matched pattern."""

    op_type: str
    inputs: list[str]
    capture_to_replace: str


class FusionPatternConfig(BaseModel):
    """Configuration for a fusion pattern."""

    pattern: NodePatternConfig
    replacement: ReplacementConfig


class ComputeCosts(BaseModel):
    """Compute costs configuration."""

    heavy_ops: list[str]
    light_ops: list[str]
    heavy_cost: int
    light_cost: int
    default_cost: int


class CostModelConfig(BaseModel):
    """Configuration for cost modeling in graph scheduling."""

    memory_sizes: dict[str, int]
    compute_costs: ComputeCosts
    compute_heavy_threshold: int
    heavy_interleave_penalty: int
    light_interleave_penalty: int


class PassConfig(BaseModel):
    """Configuration for the pass manager."""

    execution_order: list[str]
    cost_model: CostModelConfig
    fusion_patterns: dict[str, FusionPatternConfig]


class ConvergenceCriteria(BaseModel):
    """Convergence criteria for pass execution."""

    max_iterations: int = 10
    detect_cyclic_oscillation: bool = True
    require_fixpoint: bool = True


class PipelineStageModel(BaseModel):
    """Declarative specification of an optimization pipeline stage.

    Attributes:
        stage_name (str): Unique stage identifier.
        passes (list[str]): Names of compiler passes to execute in this stage.
        fixpoint_iteration (bool): Whether to iterate passes within this stage until fixpoint convergence.
        max_iterations (int): Maximum iterations when running fixpoint iteration.
    """

    stage_name: str
    passes: list[str] = Field(default_factory=list)
    fixpoint_iteration: bool = False
    max_iterations: int = 10
    model_config = ConfigDict(extra="allow")


class PassConfigModel(BaseModel):
    """Declarative configuration model for individual compiler optimization passes.

    Attributes:
        pass_name (str): Identifier name of the compiler pass.
        enabled (bool): Whether the pass is actively executed.
        prerequisites (list[str]): Passes that must execute prior to this pass.
        preserves (list[str]): Analysis/transform invariants preserved by this pass.
        options (dict[str, Union[str, int, float, bool, list[str]]]): Custom configuration knobs.
    """

    pass_name: str
    enabled: bool = True
    prerequisites: list[str] = Field(default_factory=list)
    preserves: list[str] = Field(default_factory=list)
    options: dict[str, Union[str, int, float, bool, list[str]]] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class OptLevelConfig(BaseModel):
    """Configuration for an optimization level pipeline.

    Attributes:
        execution_order (list[str]): Passes executed in this level.
        fixpoint_iteration (bool): Whether to iterate until convergence.
        max_iterations (int): Maximum iterations allowed.
    """

    execution_order: list[str] = Field(default_factory=list)
    fixpoint_iteration: bool = False
    max_iterations: int = 1


class PassAnalysisContract(BaseModel):
    """Prerequisites, consumed properties, and analysis invalidations for a compiler pass.

    Attributes:
        prerequisites (list[str]): Required passes prior to execution.
        consumed_properties (list[str]): Graph properties read by this pass.
        invalidated_analyses (list[str]): Analyses invalidated by mutations in this pass.
    """

    prerequisites: list[str] = Field(default_factory=list)
    consumed_properties: list[str] = Field(default_factory=list)
    invalidated_analyses: list[str] = Field(default_factory=list)


class PassPipelineConfig(BaseModel):
    """Declarative pass pipeline configuration.

    Attributes:
        execution_order (list[str]): Sequential pass execution order.
        convergence_criteria (ConvergenceCriteria): Fixpoint criteria.
        prerequisites (dict[str, list[str]]): Pass prerequisite dependencies.
        stages (list[PipelineStageModel]): Optimization pipeline stages.
        pass_configs (dict[str, PassConfigModel]): Pass-specific configurations.
        optimization_levels (dict[str, OptLevelConfig]): Optimization level configurations.
        pass_analyses_and_invalidation (dict[str, PassAnalysisContract]): Analysis invalidation contracts.
    """

    execution_order: list[str] = Field(default_factory=list)
    convergence_criteria: ConvergenceCriteria = Field(default_factory=ConvergenceCriteria)
    prerequisites: dict[str, list[str]] = Field(default_factory=dict)
    stages: list[PipelineStageModel] = Field(default_factory=list)
    pass_configs: dict[str, PassConfigModel] = Field(default_factory=dict)
    optimization_levels: dict[str, OptLevelConfig] = Field(default_factory=dict)
    pass_analyses_and_invalidation: dict[str, PassAnalysisContract] = Field(default_factory=dict)
    model_config = ConfigDict(extra="allow")


class RematerializationThresholds(BaseModel):
    """Thresholds for rematerialization."""

    min_memory_bytes: int
    max_compute_to_memory_ratio: float


class RematerializationRulesConfig(BaseModel):
    """Configuration for rematerialization rules."""

    target_ops: list[str]
    high_cost_ops: list[str]
    thresholds: RematerializationThresholds


class OptimizationHeuristicsConfig(BaseModel):
    """Configuration for optimization heuristics."""

    in_place_safe_ops: list[str]


class BehaviorDescriptorsConfig(BaseModel):
    """Configuration for behavior descriptors."""

    side_effect_ops: list[str]


class ObservedNodeShape(BaseModel):
    """Concrete shape and striding metadata captured for an individual IRNode.

    Attributes:
        node_id (str): Node unique identifier.
        shape (list[int]): Concrete tensor dimensions.
        dtype (str): Tensor datatype string ('float32', 'int32', etc.).
        strides (Optional[list[int]]): Memory stride layout per dimension.
        offset (Optional[int]): Memory byte offset in linear memory arena.
        byte_length (Optional[int]): Total byte size of tensor buffer.
    """

    node_id: str
    shape: list[int]
    dtype: str = "float32"
    strides: Optional[list[int]] = None
    offset: Optional[int] = None
    byte_length: Optional[int] = None


class RuntimeTensorMetadata(BaseModel):
    """Extended runtime characteristics reported by WebGPU or WASM buffers.

    Attributes:
        buffer_id (str): Low-level buffer identifier.
        element_count (int): Total number of elements.
        is_gradient (bool): Whether buffer represents a backward pass gradient.
        aliased_to (Optional[str]): Source buffer ID if aliased or reused.
        is_contiguous (bool): Contiguity flag.
    """

    buffer_id: str
    element_count: int
    is_gradient: bool = False
    aliased_to: Optional[str] = None
    is_contiguous: bool = True


class ShapeInspectionPayload(BaseModel):
    """Payload containing runtime tensor shapes observed during client browser execution.

    Attributes:
        runtime (str): Runtime execution environment ('webgpu', 'wasm_simd', etc.).
        execution_id (str): Unique identifier of dispatch execution cycle.
        observed_shapes (dict[str, list[int]]): Mapping of node IDs to observed concrete shapes.
        detailed_nodes (Optional[dict[str, ObservedNodeShape]]): Detailed node shape metrics.
        tensor_metadata (Optional[dict[str, RuntimeTensorMetadata]]): Runtime buffer metadata.
        execution_time_ms (Optional[float]): Runtime execution latency in milliseconds.
        memory_usage_bytes (Optional[int]): Total client device memory consumed.
    """

    runtime: str
    execution_id: str
    observed_shapes: dict[str, list[int]]
    detailed_nodes: Optional[dict[str, ObservedNodeShape]] = None
    tensor_metadata: Optional[dict[str, RuntimeTensorMetadata]] = None
    execution_time_ms: Optional[float] = None
    memory_usage_bytes: Optional[int] = None


class RuntimeShapePacket(BaseModel):
    """Standardized runtime shape observation packet transmitted from browser/edge runners.

    Attributes:
        packet_id (str): Unique packet identifier or execution cycle nonce.
        runtime (str): Runtime target emitting the packet ('webgpu', 'wasm_simd', 'webgl').
        graph_name (str): Identifier of the computation graph being executed.
        observations (list[ObservedNodeShape]): Concrete shape and stride measurements.
        execution_time_ms (Optional[float]): Kernel/graph execution time in milliseconds.
        memory_usage_bytes (Optional[int]): Total device/linear memory consumed.
    """

    packet_id: str
    runtime: str
    graph_name: str = "default_graph"
    observations: list[ObservedNodeShape] = Field(default_factory=list)
    execution_time_ms: Optional[float] = None
    memory_usage_bytes: Optional[int] = None
    model_config = ConfigDict(extra="allow")


class ShapeLearningProtocolConfig(BaseModel):
    """Root configuration model for shape learning protocol.

    Attributes:
        protocol_version (str): Semantic protocol version string.
        description (str): Protocol overview.
        supported_runtimes (list[str]): List of supported runtime environments.
        message_types (dict[str, dict[str, Union[str, list[str]]]]): Specification for message schemas.
    """

    protocol_version: str
    description: str
    supported_runtimes: list[str]
    message_types: dict[str, dict[str, Union[str, list[str]]]]


def load_shape_learning_protocol(path: Optional[str] = None) -> ShapeLearningProtocolConfig:
    """Load shape learning protocol configuration from YAML specification.

    Args:
        path (Optional[str]): Path to shape_learning_protocol.yaml file. Defaults to bundled YAML.

    Returns:
        ShapeLearningProtocolConfig: Validated protocol configuration model.
    """
    import os

    import yaml

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "shape_learning_protocol.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ShapeLearningProtocolConfig.model_validate(data)


class MeshPartitioningDefaultMesh(BaseModel):
    """Default device mesh grid specification.

    Attributes:
        shape (list[int]): Dimensions of the device mesh grid.
        axis_names (list[str]): Axis names (e.g., ['dp', 'tp', 'pp']).
        devices (list[int]): Physical device rank indices.
    """

    shape: list[int]
    axis_names: list[str]
    devices: list[int]


class DataParallelStrategyConfig(BaseModel):
    """Data parallel partitioning strategy configuration.

    Attributes:
        mesh_axis (str): Mesh axis name allocated for data parallelism.
        tensor_dim: Tensor dimension to shard along the mesh axis.
    """

    mesh_axis: str = "dp"
    tensor_dim: int = 0


class TensorModelParallelStrategyConfig(BaseModel):
    """Tensor and model parallel partitioning strategy configuration.

    Attributes:
        mesh_axis (str): Mesh axis name allocated for tensor parallelism.
        row_parallel_dim (int): Dimension index for row-parallel sharding.
        col_parallel_dim (int): Dimension index for column-parallel sharding.
        contracting_dim (int): Dimension index for contracting-parallel sharding.
    """

    mesh_axis: str = "tp"
    row_parallel_dim: int = 0
    col_parallel_dim: int = 1
    contracting_dim: int = -1


class PipelineParallelStrategyConfig(BaseModel):
    """Pipeline parallel partitioning strategy configuration.

    Attributes:
        mesh_axis (str): Mesh axis name allocated for pipeline parallelism.
        stage_dim (Optional[int]): Stage dimension index if applicable.
    """

    mesh_axis: str = "pp"
    stage_dim: Optional[int] = None


class MeshPartitioningStrategiesConfig(BaseModel):
    """Multi-axis partitioning strategies container.

    Attributes:
        data_parallel (DataParallelStrategyConfig): Data parallel sharding configuration.
        tensor_model_parallel (TensorModelParallelStrategyConfig): Tensor/model parallel sharding configuration.
        pipeline_parallel (PipelineParallelStrategyConfig): Pipeline parallel sharding configuration.
    """

    data_parallel: DataParallelStrategyConfig = DataParallelStrategyConfig()
    tensor_model_parallel: TensorModelParallelStrategyConfig = TensorModelParallelStrategyConfig()
    pipeline_parallel: PipelineParallelStrategyConfig = PipelineParallelStrategyConfig()


class DeviceMeshPartitioningConfig(BaseModel):
    """Declarative device mesh partitioning schema.

    Attributes:
        default_mesh (MeshPartitioningDefaultMesh): Base device grid topology.
        strategies (MeshPartitioningStrategiesConfig): Multi-axis partitioning strategies.
    """

    default_mesh: MeshPartitioningDefaultMesh
    strategies: MeshPartitioningStrategiesConfig = MeshPartitioningStrategiesConfig()


class DeviceMeshPartitioningRootConfig(BaseModel):
    """Root model for mesh_partitioning.yaml.

    Attributes:
        device_mesh_partitioning (DeviceMeshPartitioningConfig): Partitioning configuration.
    """

    device_mesh_partitioning: DeviceMeshPartitioningConfig


def load_mesh_partitioning(path: Optional[str] = None) -> DeviceMeshPartitioningConfig:
    """Load declarative mesh partitioning configuration from YAML.

    Args:
        path (Optional[str]): Path to mesh_partitioning.yaml file. Defaults to bundled YAML.

    Returns:
        DeviceMeshPartitioningConfig: Validated partitioning configuration model.
    """
    import os

    import yaml

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "spmd_mappings", "mesh_partitioning.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = DeviceMeshPartitioningRootConfig.model_validate(data)
    return root.device_mesh_partitioning


class SpmdCommunicationCondition(BaseModel):
    """Condition for SPMD communication injection.

    Attributes:
        inject (str): Operation name to inject ('AllReduce', 'AllGather', 'ReduceScatter', 'AllToAll', 'none').
        is_reduction (Optional[bool]): True if consumer is a reduction operation.
        is_grad (Optional[bool]): True if node is a backward gradient calculation.
        axes_match (Optional[bool]): True if input and node sharding axes match.
        axes_length_match (Optional[bool]): True if input and node axes counts match.
        default (Optional[bool]): True if this is the default fallback condition.
    """

    inject: str
    is_reduction: Optional[bool] = None
    is_grad: Optional[bool] = None
    axes_match: Optional[bool] = None
    axes_length_match: Optional[bool] = None
    default: Optional[bool] = None


class SpmdCommunicationRule(BaseModel):
    """Rule in SPMD communication matrix.

    Attributes:
        state (list[bool]): Boolean pair [inp_sharded, node_sharded].
        conditions (list[SpmdCommunicationCondition]): Evaluated condition rules.
    """

    state: list[bool]
    conditions: list[SpmdCommunicationCondition]


class SpmdCommunicationMatrixConfig(BaseModel):
    """Root model for communication_matrix.yaml.

    Attributes:
        communication_matrix (list[SpmdCommunicationRule]): Matrix of communication rules.
    """

    communication_matrix: list[SpmdCommunicationRule]
