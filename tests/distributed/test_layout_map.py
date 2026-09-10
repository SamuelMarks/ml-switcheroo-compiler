"""Tests for LayoutMap and ShardingSpec distributed constructs."""

from pathlib import Path
from unittest.mock import MagicMock

from ml_switcheroo_compiler.distributed.layout_map import LayoutMap, ShardingSpec


def test_sharding_spec() -> None:
    """Test ShardingSpec creation, equality, and representation."""
    s = ShardingSpec("mesh", ["x", None])
    assert s.mesh == "mesh"
    assert s.mesh_mapping == ("x", None)

    s2 = ShardingSpec("mesh", ["x", None])
    assert s == s2

    s3 = ShardingSpec("mesh", ["y", None])
    assert s != s3
    assert s != "string"

    assert repr(s) == "ShardingSpec(mesh=mesh, mapping=('x', None))"


def test_layout_map() -> None:
    """Test LayoutMap insertion, lookup, and dictionary access."""
    lm = LayoutMap("mesh")
    assert lm.device_mesh == "mesh"

    s = ShardingSpec("mesh", ["x"])
    lm.insert("layer1.*", s)

    assert lm.get("layer") is None
    assert lm.get("layer1_weight") == s

    lm["layer2"] = s
    assert lm["layer2"] == s
    assert lm["layer3"] is None

    assert repr(lm) == "LayoutMap(size=2)"


def test_layout_map_serialization_and_from_yaml(tmp_path: Path) -> None:
    """Test LayoutMap to_dict, to_yaml, and from_yaml with raw text and file paths."""
    mesh_mock = MagicMock()
    mesh_mock.name = "mesh_2d"

    lm = LayoutMap(device_mesh=mesh_mock)
    spec1 = ShardingSpec(mesh=mesh_mock, mesh_mapping=["data", None])
    spec_nomesh = ShardingSpec(mesh=None, mesh_mapping=[None, "model"])
    lm.insert("layers.0.*", spec1)
    lm.insert("layers.1.*", spec_nomesh)

    # Test to_dict and to_yaml
    data = lm.to_dict()
    assert data["version"] == "1.0"
    assert data["device_mesh"] == "mesh_2d"

    yaml_str = lm.to_yaml()
    assert "layers.0.*" in yaml_str

    # Test from_yaml with YAML string
    lm_from_str = LayoutMap.from_yaml(yaml_str, device_mesh=mesh_mock)
    assert lm_from_str.get("layers.0.weight") is not None

    # Test from_yaml with file path on disk
    yaml_file = tmp_path / "layout.yaml"
    yaml_file.write_text(yaml_str, encoding="utf-8")

    lm_from_file = LayoutMap.from_yaml(str(yaml_file), device_mesh=mesh_mock)
    assert lm_from_file.get("layers.0.weight") is not None

    # Test LayoutMap with device_mesh=None to_dict branch
    lm_none = LayoutMap(device_mesh=None)
    assert lm_none.to_dict()["device_mesh"] is None
