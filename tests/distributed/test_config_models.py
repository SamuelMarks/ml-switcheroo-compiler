"""Unit tests for distributed topology and device mesh configuration models."""

import os

import yaml

from ml_switcheroo_compiler.distributed.config_models import (
    ClusterMeshConfig,
    CommunicationTopologyConfig,
    DeviceMeshYamlConfig,
    MeshMappingConfig,
    MicrobatchSplittingConfig,
    PipelineTopologiesConfig,
    StageCommunicationConfig,
)


def test_pipeline_config_model() -> None:
    """Test pipeline topologies configuration validation and accessors."""
    data = {
        "default": {
            "microbatch_splitting": {"num_microbatches": 4, "strategy": "chunk"},
            "mesh_mapping": {"devices_per_stage": 1},
            "stage_communication": {"protocol": "p2p_queue"},
        }
    }
    config = PipelineTopologiesConfig(root=data)
    default_config = config.get("default")
    assert default_config is not None
    assert default_config.microbatch_splitting.num_microbatches == 4
    assert default_config.stage_communication.protocol == "p2p_queue"

    assert list(config.items())[0][0] == "default"
    assert config.model_dump()["default"]["mesh_mapping"]["devices_per_stage"] == 1
    assert config.dict()["default"]["mesh_mapping"]["devices_per_stage"] == 1
    assert config.get("missing") is None

    assert MicrobatchSplittingConfig(num_microbatches=1, strategy="strat").num_microbatches == 1
    assert MeshMappingConfig(devices_per_stage=2).devices_per_stage == 2
    assert StageCommunicationConfig(protocol="p").protocol == "p"


def test_pipeline_topologies_yaml_file_validation() -> None:
    """Validate that pipeline_topologies.yaml conforms strictly to PipelineTopologiesConfig schema."""
    yaml_path = os.path.join(os.path.dirname(__file__), "../../src/ml_switcheroo_compiler/distributed/pipeline_topologies.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    validated = PipelineTopologiesConfig.model_validate(raw_data)
    assert "default" in validated.root
    assert "1f1b" in validated.root
    assert "gpipe" in validated.root
    assert "pipedream" in validated.root
    assert "webrtc_pipeline" in validated.root
    assert "zero_bubble" in validated.root


def test_device_mesh_yaml_file_validation() -> None:
    """Validate that device_mesh.yaml conforms strictly to DeviceMeshYamlConfig schema."""
    yaml_path = os.path.join(os.path.dirname(__file__), "../../src/ml_switcheroo_compiler/distributed/device_mesh.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    validated = DeviceMeshYamlConfig.model_validate(raw_data)
    assert validated.version == "1.0"
    assert "default" in validated.cluster_meshes
    assert "mesh_2d_2x2" in validated.cluster_meshes
    assert "ring" in validated.communication_topologies

    cm = ClusterMeshConfig(shape=[2, 2], axis_names=["x", "y"], devices=[0, 1, 2, 3], topology="grid")
    assert cm.shape == [2, 2]

    ct = CommunicationTopologyConfig(type="ring", description="cyclic ring")
    assert ct.type == "ring"


def test_consolidated_distributed_topology_models() -> None:
    """Test instantiation and validation of consolidated topology models."""
    from ml_switcheroo_compiler.distributed.config_models import (
        ClusterTopologyConfig,
        CollectiveAlgorithmConfig,
        DeviceMeshGridConfig,
        DistributedTopologyConfig,
        InterconnectBandwidthConfig,
    )

    ib = InterconnectBandwidthConfig(bus_type="nvlink", bandwidth_gbps=900.0, latency_us=0.5, bidirectional=True)
    assert ib.bus_type == "nvlink"
    assert ib.bandwidth_gbps == 900.0

    cluster = ClusterTopologyConfig(num_nodes=2, devices_per_node=8, interconnect=ib, network_type="nvlink_mesh")
    assert cluster.num_nodes == 2

    alg = CollectiveAlgorithmConfig(collective_op="AllReduce", algorithm="ring", topology_type="ring")
    assert alg.collective_op == "AllReduce"

    grid = DeviceMeshGridConfig(shape=[2, 2], axis_names=["dp", "tp"], devices=[0, 1, 2, 3], topology="grid")
    assert grid.shape == [2, 2]

    dist_cfg = DistributedTopologyConfig(
        version="1.0",
        cluster_topologies={"c1": cluster},
        device_mesh_grids={"g1": grid},
        collective_algorithms=[alg],
    )
    assert "c1" in dist_cfg.cluster_topologies
    assert "g1" in dist_cfg.device_mesh_grids
    assert len(dist_cfg.collective_algorithms) == 1


def test_load_consolidated_topology_file() -> None:
    """Test load_consolidated_topology against distributed_topology.yaml."""
    from ml_switcheroo_compiler.distributed.config_models import load_consolidated_topology

    cfg = load_consolidated_topology()
    assert cfg.version == "1.0"
    assert "h100_nvlink_8gpu" in cfg.cluster_topologies
    assert "mesh_2d_2x2" in cfg.device_mesh_grids
    assert any(a.collective_op == "AllReduce" for a in cfg.collective_algorithms)
    assert "default" in cfg.pipeline_topologies
    assert cfg.webrtc_topology is not None


def test_load_consolidated_topology_fallback(tmp_path: object) -> None:
    """Test load_consolidated_topology fallback when distributed_topology.yaml is missing."""
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.config_models import load_consolidated_topology

    # Simulate missing distributed_topology.yaml so it consolidates from sibling files
    real_exists = os.path.exists

    def mock_exists(path: str) -> bool:
        if path.endswith("distributed_topology.yaml"):
            return False
        return real_exists(path)

    with patch("os.path.exists", side_effect=mock_exists):
        cfg = load_consolidated_topology()
        assert cfg.version == "1.0"
        assert "default" in cfg.device_mesh_grids
        assert "default" in cfg.pipeline_topologies
        assert cfg.webrtc_topology is not None

    # Simulate all files missing
    with patch("os.path.exists", return_value=False):
        cfg_empty = load_consolidated_topology()
        assert cfg_empty.version == "1.0"
        assert cfg_empty.device_mesh_grids == {}
        assert cfg_empty.pipeline_topologies == {}
        assert cfg_empty.webrtc_topology is None


def test_load_distributed_topologies(tmp_path: object) -> None:
    """Test load_distributed_topologies loader function."""
    from ml_switcheroo_compiler.distributed.config_models import load_distributed_topologies

    # Default path
    cfg = load_distributed_topologies()
    assert cfg is not None
    assert cfg.cluster_meshes is not None

    # Custom path
    yaml_path = os.path.join(os.path.dirname(__file__), "../../src/ml_switcheroo_compiler/distributed/distributed_topologies.yaml")
    cfg_custom = load_distributed_topologies(path=yaml_path)
    assert cfg_custom is not None
