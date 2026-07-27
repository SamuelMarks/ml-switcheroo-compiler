"""Test module."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.distributed import DataParallel, Distribution, ModelParallel, TensorLayout, distribute_tensor, distribution, initialize, list_devices, set_distribution


def test_distributed_init():
    dist = Distribution(device_mesh="mesh")
    assert dist.device_mesh == "mesh"

    with dist.scope():
        assert distribution() is dist

    dp = DataParallel()
    assert dp.device_mesh is None

    mp = ModelParallel(layout_map="lm")
    assert mp.layout_map == "lm"

    tl = TensorLayout(("a",))
    assert tl.axes == ("a",)

    tl2 = TensorLayout(axes=("b",))
    assert tl2.axes == ("b",)

    initialize()

    devs = list_devices()
    assert isinstance(devs, list)

    set_distribution(dp)
    assert distribution() is dp

    set_distribution(None)

    # distribute_tensor no dist
    assert distribute_tensor(1) == 1
    assert distribute_tensor(tensor=2) == 2

    # distribute_tensor with dist eager
    set_distribution(dp)
    config.eager_mode = True
    assert distribute_tensor(3) == 3

    # distribute_tensor tracing
    config.eager_mode = False
    import ml_switcheroo_compiler.tracing.state as state

    state.global_tracing_state.is_tracing = False

    try:
        distribute_tensor(4)
    except Exception:
        pass


def test_dist_scope():
    from ml_switcheroo_compiler.distributed import Distribution

    dist = Distribution()
    with dist.scope():
        pass


def test_distributed_initialize_fallback(mocker):
    from ml_switcheroo_compiler.distributed import initialize

    class MockBackend:
        pass

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend())
    initialize()

    class MockBackend2:
        def initialize_distributed(self, *args, **kwargs):
            pass

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackend2())
    initialize()

    mocker.patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("Failed"))
    initialize()
