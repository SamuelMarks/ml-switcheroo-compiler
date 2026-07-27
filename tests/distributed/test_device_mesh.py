"""Test module."""

import pytest

from ml_switcheroo_compiler.distributed.device_mesh import DeviceMesh


def test_device_mesh():
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
