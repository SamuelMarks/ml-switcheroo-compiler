"""Unit tests for DeviceMesh representation and topology mapping."""

import pytest

from ml_switcheroo_compiler.distributed.device_mesh import DeviceMesh


def test_device_mesh_basic() -> None:
    """Test basic instantiation, equality, and repr of DeviceMesh."""
    dm = DeviceMesh([2], ["x"])
    assert dm.shape == (2,)
    assert dm.axis_names == ("x",)
    assert dm.devices == (0, 1)

    dm2 = DeviceMesh([2], ["x"], [10, 11])
    assert dm2.devices == (10, 11)

    with pytest.raises(ValueError):
        DeviceMesh([2], ["x", "y"])

    with pytest.raises(ValueError):
        DeviceMesh([2], ["x"], [10])

    assert repr(dm) == "DeviceMesh(shape=(2,), axis_names=('x',))"

    assert dm == DeviceMesh([2], ["x"])
    assert dm != dm2
    assert dm != "string"


def test_device_mesh_from_yaml() -> None:
    """Test loading meshes from declarative device_mesh.yaml."""
    dm_default = DeviceMesh.from_yaml("default")
    assert dm_default.shape == (4,)
    assert dm_default.axis_names == ("data",)
    assert dm_default.devices == (0, 1, 2, 3)

    dm_2d = DeviceMesh.from_yaml("mesh_2d_2x2")
    assert dm_2d.shape == (2, 2)
    assert dm_2d.axis_names == ("data", "model")
    assert dm_2d.devices == (0, 1, 2, 3)

    with pytest.raises(KeyError):
        DeviceMesh.from_yaml("non_existent_mesh_xyz")


def test_device_mesh_coordinate_mapping() -> None:
    """Test coordinate conversion to and from device ranks."""
    dm = DeviceMesh([2, 3], ["row", "col"])
    # Rank coordinates
    assert dm.get_coords(0) == (0, 0)
    assert dm.get_coords(1) == (0, 1)
    assert dm.get_coords(2) == (0, 2)
    assert dm.get_coords(3) == (1, 0)
    assert dm.get_coords(5) == (1, 2)

    # Coords to rank
    assert dm.get_rank((0, 0)) == 0
    assert dm.get_rank((0, 2)) == 2
    assert dm.get_rank((1, 2)) == 5

    # Errors
    with pytest.raises(ValueError):
        dm.get_coords(99)
    with pytest.raises(ValueError):
        dm.get_rank((0,))
    with pytest.raises(ValueError):
        dm.get_rank((0, 99))


def test_device_mesh_neighbors_and_serialization() -> None:
    """Test neighbor traversal along axes and dictionary serialization."""
    dm = DeviceMesh([2, 2], ["data", "model"])
    # rank 0 coords (0, 0). Along 'data': prev is (1, 0) = rank 2, next is (1, 0) = rank 2
    prev_r, next_r = dm.get_neighbors(0, "data")
    assert prev_r == 2
    assert next_r == 2

    # Along 'model': prev is (0, 1) = rank 1, next is (0, 1) = rank 1
    prev_m, next_m = dm.get_neighbors(0, "model")
    assert prev_m == 1
    assert next_m == 1

    with pytest.raises(ValueError):
        dm.get_neighbors(0, "invalid_axis")

    d = dm.to_dict()
    assert d["shape"] == (2, 2)
    assert d["axis_names"] == ("data", "model")
    assert d["devices"] == (0, 1, 2, 3)
