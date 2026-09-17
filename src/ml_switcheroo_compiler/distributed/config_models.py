"""Pydantic models for distributed topologies and device meshes."""

import os
from collections.abc import ItemsView
from typing import Optional

import yaml
from pydantic import BaseModel, Field, RootModel


class MicrobatchSplittingConfig(BaseModel):
    """Configuration for splitting microbatches."""

    num_microbatches: int
    strategy: str


class MeshMappingConfig(BaseModel):
    """Configuration for mapping stages to devices."""

    devices_per_stage: int


class StageCommunicationConfig(BaseModel):
    """Configuration for communication between pipeline stages."""

    protocol: str


class DependencyConfig(BaseModel):
    """Configuration for synchronization dependencies."""

    source_stage: str
    target_stage: str
    offset_mb: int


class SchedulePhaseConfig(BaseModel):
    """Configuration for a schedule phase."""

    type: str
    operations: list[str]
    count_expression: str  # e.g. "num_stages - 1" or "num_microbatches - num_stages + 1"


class ScheduleConfig(BaseModel):
    """Configuration for a pipeline schedule."""

    phases: list[SchedulePhaseConfig]


class TopologyConfig(BaseModel):
    """Configuration for a specific pipeline topology."""

    microbatch_splitting: MicrobatchSplittingConfig
    mesh_mapping: MeshMappingConfig
    stage_communication: StageCommunicationConfig
    dependencies: list[DependencyConfig] = []
    schedule: Optional[ScheduleConfig] = None


class PipelineTopologiesConfig(RootModel[dict[str, TopologyConfig]]):
    """Configuration for all pipeline topologies."""

    root: dict[str, TopologyConfig]

    def dict(self, *args: object, **kwargs: object) -> dict[str, object]:
        """Return dict representation.

        Args:
            *args (object): Positional dump args.
            **kwargs (object): Keyword dump args.

        Returns:
            dict[str, object]: Dictionary representation.
        """
        return super().model_dump(*args, **kwargs)

    def items(self) -> ItemsView[str, TopologyConfig]:
        """Return items from the underlying dictionary.

        Returns:
            ItemsView[str, TopologyConfig]: Items view of mapping.
        """
        return self.root.items()

    def get(self, key: str, default: Optional[TopologyConfig] = None) -> Optional[TopologyConfig]:
        """Get topology config by key.

        Args:
            key (str): The configuration key.
            default (Optional[TopologyConfig]): Default value if key is not found.

        Returns:
            Optional[TopologyConfig]: The topology config or default.
        """
        return self.root.get(key, default)


class ClusterMeshConfig(BaseModel):
    """Configuration for a cluster mesh."""

    shape: list[int]
    axis_names: list[str]
    devices: list[int]
    topology: str


class CommunicationTopologyConfig(BaseModel):
    """Configuration for a communication topology pattern."""

    type: str
    description: str


class InterconnectBandwidthConfig(BaseModel):
    """Configuration for node and device interconnect bandwidth and latency."""

    bus_type: str = "nvlink"
    bandwidth_gbps: float = 100.0
    latency_us: float = 1.0
    bidirectional: bool = True


class ClusterTopologyConfig(BaseModel):
    """Configuration for physical cluster topology."""

    num_nodes: int = 1
    devices_per_node: int = 8
    interconnect: InterconnectBandwidthConfig = InterconnectBandwidthConfig()
    network_type: str = "cluster"
    description: Optional[str] = None


class CollectiveAlgorithmConfig(BaseModel):
    """Configuration mapping collective operations to algorithms and topologies."""

    collective_op: str
    algorithm: str
    min_bytes: int = 0
    max_bytes: Optional[int] = None
    topology_type: str = "ring"


class DeviceMeshGridConfig(BaseModel):
    """Configuration for device mesh grids."""

    shape: list[int]
    axis_names: list[str]
    devices: list[int]
    topology: str
    interconnect_bandwidth_gbps: Optional[float] = None
    collective_algorithms: dict[str, str] = {}


class DistributedTopologyConfig(BaseModel):
    """Consolidated schema for cluster, mesh grids, collectives, pipelines, and WebRTC."""

    version: str = "1.0"
    cluster_topologies: dict[str, ClusterTopologyConfig] = {}
    device_mesh_grids: dict[str, DeviceMeshGridConfig] = {}
    collective_algorithms: list[CollectiveAlgorithmConfig] = []
    pipeline_topologies: dict[str, TopologyConfig] = {}
    webrtc_topology: Optional[dict[str, object]] = None


class DeviceMeshYamlConfig(BaseModel):
    """Root configuration schema for device_mesh.yaml."""

    version: str
    cluster_meshes: dict[str, ClusterMeshConfig]
    communication_topologies: dict[str, CommunicationTopologyConfig]


class ShardingSpecConfig(BaseModel):
    """Configuration schema for a tensor sharding specification.

    Attributes:
        mesh_name (Optional[str]): Target device mesh name.
        mesh_mapping (list[Optional[str]]): Sharded axes or None for replicated.
    """

    mesh_name: Optional[str] = None
    mesh_mapping: list[Optional[str]] = []


class LayoutMapYamlConfig(BaseModel):
    """Configuration schema for declarative layout map YAML definitions.

    Attributes:
        version (str): Configuration version string.
        device_mesh (Optional[str]): Associated cluster device mesh name.
        specs (dict[str, ShardingSpecConfig]): Tensor path regex mapped to sharding configs.
    """

    version: str = "1.0"
    device_mesh: Optional[str] = None
    specs: dict[str, ShardingSpecConfig] = {}


def load_consolidated_topology(yaml_path: Optional[str] = None) -> DistributedTopologyConfig:
    """Load and validate consolidated distributed topology schema.

    If yaml_path is omitted, loads distributed_topology.yaml if present, or
    consolidates device_mesh.yaml, pipeline_topologies.yaml, and webrtc_topology.yaml.

    Args:
        yaml_path (Optional[str]): Explicit path to a consolidated distributed topology YAML file.

    Returns:
        DistributedTopologyConfig: Validated consolidated topology configuration.
    """
    base_dir = os.path.dirname(__file__)
    target_path = yaml_path or os.path.join(base_dir, "distributed_topology.yaml")

    if os.path.exists(target_path):
        with open(target_path, encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        return DistributedTopologyConfig.model_validate(raw_data)

    # Fallback consolidation from sibling YAML files
    consolidated_dict: dict[str, object] = {
        "version": "1.0",
        "cluster_topologies": {},
        "device_mesh_grids": {},
        "collective_algorithms": [],
        "pipeline_topologies": {},
        "webrtc_topology": None,
    }

    mesh_path = os.path.join(base_dir, "device_mesh.yaml")
    if os.path.exists(mesh_path):
        with open(mesh_path, encoding="utf-8") as f:
            mesh_raw = yaml.safe_load(f) or {}
        mesh_cfg = DeviceMeshYamlConfig.model_validate(mesh_raw)
        grids: dict[str, object] = {}
        for name, cm in mesh_cfg.cluster_meshes.items():
            grids[name] = {
                "shape": cm.shape,
                "axis_names": cm.axis_names,
                "devices": cm.devices,
                "topology": cm.topology,
            }
        consolidated_dict["device_mesh_grids"] = grids

    pipe_path = os.path.join(base_dir, "pipeline_topologies.yaml")
    if os.path.exists(pipe_path):
        with open(pipe_path, encoding="utf-8") as f:
            pipe_raw = yaml.safe_load(f) or {}
        pipe_cfg = PipelineTopologiesConfig.model_validate(pipe_raw)
        consolidated_dict["pipeline_topologies"] = pipe_cfg.root

    webrtc_path = os.path.join(base_dir, "webrtc_topology.yaml")
    if os.path.exists(webrtc_path):
        with open(webrtc_path, encoding="utf-8") as f:
            webrtc_raw = yaml.safe_load(f) or {}
        consolidated_dict["webrtc_topology"] = webrtc_raw

    return DistributedTopologyConfig.model_validate(consolidated_dict)


class ClusterMeshSpec(BaseModel):
    """Declarative specification for a distributed cluster mesh.

    Attributes:
        mesh_name (str): Mesh identifier.
        shape (list[int]): Dimensions of the device mesh grid.
        axis_names (list[str]): Names of the mesh axes (e.g. ['dp', 'tp', 'pp']).
        devices (list[int]): Physical device rank indices.
        topology (str): Interconnect topology ('torus', 'ring', 'tree', 'all_to_all').
    """

    mesh_name: str
    shape: list[int]
    axis_names: list[str]
    devices: list[int]
    topology: str = "ring"


class DeviceTopologySpec(BaseModel):
    """Declarative specification for a cluster host device topology.

    Attributes:
        host_id (str): Host identifier.
        device_count (int): Number of accelerator devices.
        accelerator_type (str): Accelerator type ('cuda', 'rocm', 'tpu', 'wasm', 'webgpu').
        interconnect_bandwidth_gbps (float): Peak host/device interconnect bandwidth in GB/s.
    """

    host_id: str
    device_count: int
    accelerator_type: str = "cuda"
    interconnect_bandwidth_gbps: float = 900.0


class SignalingTopologySpec(BaseModel):
    """Declarative specification for WebRTC signaling topologies and peer coordination.

    Attributes:
        signaling_url (str): WebSocket or HTTP signaling server endpoint.
        ice_servers (list[str]): STUN/TURN server URLs.
        high_watermark_bytes (int): Buffer high-water mark for backpressure management.
        low_watermark_bytes (int): Buffer low-water mark for resumed data transmission.
        chunk_size_bytes (int): Fragmentation chunk size for large tensor transfers.
    """

    signaling_url: str = "ws://localhost:8080"
    ice_servers: list[str] = Field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    high_watermark_bytes: int = 1048576
    low_watermark_bytes: int = 262144
    chunk_size_bytes: int = 65536


class CommunicationCostMatrixSpec(BaseModel):
    """Declarative specification for pairwise communication latency and bandwidth cost matrices.

    Attributes:
        matrix_name (str): Identifier for cost matrix.
        device_ids (list[int]): List of device IDs corresponding to matrix rows/columns.
        latency_matrix_us (list[list[float]]): NxN matrix of interconnect latency in microseconds.
        bandwidth_matrix_gbps (list[list[float]]): NxN matrix of interconnect bandwidth in GB/s.
    """

    matrix_name: str
    device_ids: list[int]
    latency_matrix_us: list[list[float]]
    bandwidth_matrix_gbps: list[list[float]]


class DistributedTopologiesConfig(BaseModel):
    """Container for declarative cluster, device, and signaling topologies.

    Attributes:
        cluster_meshes (dict[str, ClusterMeshSpec]): Map of cluster mesh names to specs.
        device_topologies (dict[str, DeviceTopologySpec]): Map of host IDs to device specs.
        signaling_topologies (dict[str, SignalingTopologySpec]): Map of signaling setups.
        communication_cost_matrices (dict[str, CommunicationCostMatrixSpec]): Map of cost matrices.
    """

    cluster_meshes: dict[str, ClusterMeshSpec] = Field(default_factory=dict)
    device_topologies: dict[str, DeviceTopologySpec] = Field(default_factory=dict)
    signaling_topologies: dict[str, SignalingTopologySpec] = Field(default_factory=dict)
    communication_cost_matrices: dict[str, CommunicationCostMatrixSpec] = Field(default_factory=dict)


class DistributedTopologiesRootConfig(BaseModel):
    """Root model for distributed_topologies.yaml.

    Attributes:
        distributed_topologies (DistributedTopologiesConfig): The topologies configuration.
    """

    distributed_topologies: DistributedTopologiesConfig


def load_cluster_topology(path: Optional[str] = None) -> DeviceMeshYamlConfig:
    """Load and validate declarative cluster topology specification from device_mesh.yaml.

    Args:
        path (Optional[str]): Optional custom path to device_mesh.yaml.

    Returns:
        DeviceMeshYamlConfig: Validated device mesh YAML configuration.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_mesh.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return DeviceMeshYamlConfig.model_validate(data)


def load_distributed_topologies(path: Optional[str] = None) -> DistributedTopologiesConfig:
    """Load and validate declarative distributed cluster topologies from YAML.

    Args:
        path (Optional[str]): Optional path to distributed_topologies.yaml.

    Returns:
        DistributedTopologiesConfig: Validated distributed topologies configuration.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "distributed_topologies.yaml")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = DistributedTopologiesRootConfig.model_validate(data)
    return root.distributed_topologies
