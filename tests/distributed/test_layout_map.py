"""Test module."""

from ml_switcheroo_compiler.distributed.layout_map import LayoutMap, ShardingSpec


def test_sharding_spec():
    s = ShardingSpec("mesh", ["x", None])
    assert s.mesh == "mesh"
    assert s.mesh_mapping == ("x", None)

    s2 = ShardingSpec("mesh", ["x", None])
    assert s == s2

    s3 = ShardingSpec("mesh", ["y", None])
    assert s != s3
    assert s != "string"

    assert repr(s) == "ShardingSpec(mesh=mesh, mapping=('x', None))"


def test_layout_map():
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
