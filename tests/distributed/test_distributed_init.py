import pytest


def test_distributed_init():
    import ml_switcheroo_compiler.distributed as dist
    from ml_switcheroo_compiler.distributed import DeviceMesh, LayoutMap

    mesh = DeviceMesh((1,), ("x",))
    d = dist.Distribution(device_mesh=mesh)
    assert d.device_mesh == mesh

    with d.scope():
        assert dist.distribution() == d

    dp = dist.DataParallel(device_mesh=mesh)
    assert dp.device_mesh == mesh

    lm = LayoutMap()
    mp = dist.ModelParallel(layout_map=lm)
    assert mp.layout_map == lm

    tl = dist.TensorLayout(axes=("batch", "features"))
    assert tl.axes == ("batch", "features")

    tl2 = dist.TensorLayout(axes=("batch",))
    assert tl2.axes == ("batch",)

    import unittest.mock

    with unittest.mock.patch("ml_switcheroo_compiler.core.device.get_physical_devices", return_value=["device"]):
        devices = dist.list_devices()
        assert devices == ["device"]

    dist.set_distribution(dp)
    assert dist.distribution() == dp

    with unittest.mock.patch("ml_switcheroo_compiler.ops.distributed_ops.shard_tensor", return_value="sharded"):
        res = dist.distribute_tensor(1)
        assert res == "sharded"


def test_distributed_init_eager(monkeypatch):
    import ml_switcheroo_compiler.distributed as dist
    from ml_switcheroo_compiler.core import config

    monkeypatch.setattr(config, "eager_mode", True)
    dist.set_distribution(dist.Distribution())
    assert dist.distribute_tensor(1) == 1
    import ml_switcheroo_compiler.distributed as dist
    from ml_switcheroo_compiler.core import config

    monkeypatch.setattr(config, "eager_mode", False)

    # mock shard_tensor
    def mock_shard_tensor(*args, **kwargs):
        return "sharded"

    monkeypatch.setattr("ml_switcheroo_compiler.ops.distributed_ops.shard_tensor", mock_shard_tensor)

    dist.set_distribution(dist.Distribution())
    assert dist.distribute_tensor(1) == "sharded"


def test_distributed_init_no_dist():
    import ml_switcheroo_compiler.distributed as dist

    dist.set_distribution(None)
    assert dist.distribute_tensor(2) == 2


def test_distributed_initialize(monkeypatch):
    import ml_switcheroo_compiler.distributed as dist
    from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

    class MockBackendWithInit:
        def initialize_distributed(self):
            self.called = True

    mock_backend = MockBackendWithInit()
    monkeypatch.setattr("ml_switcheroo_compiler.backends.registry.get_active_backend", lambda: mock_backend)

    dist.initialize()
    assert mock_backend.called

    class MockBackendNoInit:
        pass

    mock_backend2 = MockBackendNoInit()
    monkeypatch.setattr("ml_switcheroo_compiler.backends.registry.get_active_backend", lambda: mock_backend2)
    with pytest.raises(BackendNotSupportedError):
        dist.initialize()
