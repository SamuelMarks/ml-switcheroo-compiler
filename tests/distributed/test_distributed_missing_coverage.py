from unittest.mock import patch

import pytest

import ml_switcheroo_compiler.core.config as config
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.distributed import DataParallel, Distribution, ModelParallel, TensorLayout, TensorLayoutClass, distribute_tensor, distribution, initialize, list_devices, set_distribution


class DummyBackendNoSupport:
    __name__ = "dummy"


def test_initialize_missing_support():
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackendNoSupport()):
        with pytest.raises(BackendNotSupportedError, match="does not support initialize_distributed"):
            initialize()


class DummyBackendSupport:
    __name__ = "dummy"

    def initialize_distributed(self, *args, **kwargs):
        pass


def test_initialize_with_support():
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackendSupport()):
        initialize()


def test_distribution_classes():
    d = Distribution(device_mesh=1)
    assert d.device_mesh == 1

    with d.scope():
        assert distribution() is d

    assert distribution() is None

    dp = DataParallel(device_mesh=2)
    assert dp.device_mesh == 2

    mp = ModelParallel(layout_map=3)
    assert mp.layout_map == 3

    tl = TensorLayout(axes=(1, 2))
    assert isinstance(tl, TensorLayoutClass)
    assert tl.axes == (1, 2)

    tl2 = TensorLayout((3, 4))
    assert tl2.axes == (3, 4)

    tl3 = TensorLayout()
    assert tl3.axes == ()


def test_list_devices():
    with patch("ml_switcheroo_compiler.core.device.get_physical_devices", return_value=["dev1"]):
        assert list_devices() == ["dev1"]


def test_distribute_tensor():
    original_dist = distribution()
    try:
        # None dist
        set_distribution(None)
        assert distribute_tensor(1) == 1
        assert distribute_tensor(tensor=2) == 2

        # with dist and eager mode
        d = Distribution()
        set_distribution(d)
        config.eager_mode = True
        assert distribute_tensor(3) == 3

        # with dist and not eager
        config.eager_mode = False
        with patch("ml_switcheroo_compiler.ops.distributed_ops.shard_tensor", return_value="sharded"):
            assert distribute_tensor(4) == "sharded"
    finally:
        set_distribution(original_dist)
        config.eager_mode = False  # restore default
