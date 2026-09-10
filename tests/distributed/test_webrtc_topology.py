"""Unit tests for WebRTC topology YAML validation."""

import os

import yaml

from ml_switcheroo_compiler.backends.edge.config_models import WebrtcTopologyConfig


def test_webrtc_topology() -> None:
    """Validate webrtc_topology.yaml against WebrtcTopologyConfig schema."""
    yaml_path = os.path.join(os.path.dirname(__file__), "../../src/ml_switcheroo_compiler/distributed/webrtc_topology.yaml")
    assert os.path.exists(yaml_path)

    with open(yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    validated = WebrtcTopologyConfig(**raw)
    assert len(validated.webrtc_config.ice_servers) >= 1
    assert "init_peer_connection" in validated.templates
    assert "allreduce_emit" in validated.templates
    assert "allgather_emit" in validated.templates
    assert "alltoall_emit" in validated.templates
    assert "reducescatter_emit" in validated.templates
