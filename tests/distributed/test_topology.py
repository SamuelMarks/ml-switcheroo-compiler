import pytest

from ml_switcheroo_compiler.distributed import DeviceMesh, LayoutMap, ShardingSpec


def test_device_mesh():
    mesh = DeviceMesh(shape=(2, 4), axis_names=("data", "model"))
    assert mesh.shape == (2, 4)
    assert mesh.axis_names == ("data", "model")
    assert len(mesh.devices) == 8

    with pytest.raises(ValueError, match="Length of shape and axis_names must match"):
        DeviceMesh(shape=(2, 4), axis_names=("data",))

    with pytest.raises(ValueError, match="Expected 8 devices, got 4"):
        DeviceMesh(shape=(2, 4), axis_names=("data", "model"), devices=tuple(range(4)))


def test_layout_map():
    mesh = DeviceMesh(shape=(2, 4), axis_names=("data", "model"))
    spec1 = ShardingSpec(mesh, ("data", None))
    spec2 = ShardingSpec(mesh, (None, "model"))

    layout = LayoutMap()
    layout.insert("layer1/weights", spec1)
    layout.insert("layer2/.*", spec2)

    assert layout.get("layer1/weights") == spec1
    assert layout.get("layer2/weights") == spec2
    assert layout.get("layer3/weights") is None
