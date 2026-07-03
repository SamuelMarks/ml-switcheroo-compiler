"""Module docstring."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.distributed.dummy import (
    _dummy_all_reduce,
    _dummy_reduce_scatter,
)
from ml_switcheroo_compiler.distributed import DeviceMesh, LayoutMap, ShardingSpec


def test_dummy_extras() -> object:
    """Function docstring."""
    tensor = np.ones(5)
    assert _dummy_reduce_scatter(tensor, "sum", 0, None) is tensor
    assert _dummy_all_reduce(tensor, "sum", None) is tensor


def test_device_mesh_repr_and_eq() -> object:
    """Function docstring."""
    mesh1 = DeviceMesh(shape=(2,), axis_names=("data",))
    mesh2 = DeviceMesh(shape=(2,), axis_names=("data",))
    mesh3 = DeviceMesh(shape=(2,), axis_names=("model",))

    assert mesh1 == mesh2
    assert mesh1 != mesh3
    assert mesh1 != "not_a_mesh"
    assert "DeviceMesh" in repr(mesh1)


def test_layout_map_repr_and_eq() -> object:
    """Function docstring."""
    mesh = DeviceMesh(shape=(2,), axis_names=("data",))
    spec1 = ShardingSpec(mesh, ("data",))
    spec2 = ShardingSpec(mesh, ("data",))
    spec3 = ShardingSpec(mesh, (None,))

    assert spec1 == spec2
    assert spec1 != spec3
    assert spec1 != "not_a_spec"
    assert "ShardingSpec" in repr(spec1)

    layout = LayoutMap()
    assert "LayoutMap" in repr(layout)


def test_device_mesh_explicit_devices() -> object:
    """Function docstring."""
    mesh = DeviceMesh(shape=(2,), axis_names=("data",), devices=(0, 1))
    assert mesh.devices == (0, 1)
