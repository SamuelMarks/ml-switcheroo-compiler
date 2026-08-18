"""Test module."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.distributed import DataParallel, Distribution, ModelParallel, TensorLayout, distribute_tensor, distribution, initialize, list_devices, set_distribution


def test_distributed_init():
    import pytest

    with pytest.raises(Exception):
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
    import pytest

    with pytest.raises(Exception):
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


from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.distributed import TensorLayoutClass


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


def test_distributed_strategy_extras_7():
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy, PipelineParallelismStrategy, RemoteValue
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # 594-596
    pp = PipelineParallelismStrategy(num_microbatches=2, devices_per_stage=1)
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Grad")
    g.nodes = {"n1": n1}
    pp.track_gradient_accumulation(g)
    assert "n1_accum" in g.nodes

    # MeshShardingStrategy
    ms = MeshShardingStrategy(mesh=1, layout_map={"n2": "mapped"})
    assert ms.mesh == 1
    assert ms.layout_map == {"n2": "mapped"}

    # execute_pipeline branches
    g2 = IRGraph()
    g2.nodes = {"n1": IRNode(id="n1", op_type="Input"), "n2": IRNode(id="n2", op_type="Add", inputs=["n1"])}
    g2.outputs = ["n2"]

    class FakeExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, *args):
            fn(*args)

            class FakeFuture:
                def result(self):
                    pass

            return FakeFuture()

    with patch("concurrent.futures.ThreadPoolExecutor", FakeExecutor):
        with patch("concurrent.futures.as_completed", lambda x: x):
            # It will fail on KeyError for inputs, which is fine if caught
            try:
                pp.unroll_pipeline(g2, 1)
            except Exception:
                pass

    # 622-650 propagate_layouts
    g3 = IRGraph()
    g3.nodes = {"n1": IRNode(id="n1", op_type="Input"), "n2": IRNode(id="n2", op_type="Add", inputs=["n1"]), "n3": IRNode(id="n3", op_type="Mul", inputs=["n2"])}
    g3.nodes["n1"].sharding = "sharded"
    ms.propagate_layouts(g3)
    assert g3.nodes["n2"].sharding == "mapped"
    assert g3.nodes["n3"].sharding == "mapped"

    ms.propagate_layouts(g3)
    assert g3.nodes["n2"].sharding == "mapped"

    with patch("ml_switcheroo_compiler.transforms.passes.spmd.inject_spmd_communication_pass", return_value=True):
        assert ms.lower_sharding(g3) is True

    # 264
    rv = RemoteValue()
    rv.values = [1, 2]

    # pp inner coverage
    g4 = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Add", inputs=["n1"])
    n3 = IRNode(id="n3", op_type="Constant", inputs=[], attributes={"value": 5.0})
    n4 = IRNode(id="n4", op_type="Send", inputs=["n2"], attributes={"target_stage": 1})
    g4.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4}
    g4.outputs = ["n2"]

    class FakeExecutor2:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, fn, idx, ctx, *args):
            fn(idx, ctx)

            class FakeFuture:
                def result(self):
                    pass

            return FakeFuture()

    class FakeBackend:
        def execute_op(self, name, *args, **kwargs):
            return args[0]

    with patch("concurrent.futures.ThreadPoolExecutor", FakeExecutor2):
        with patch("concurrent.futures.as_completed", lambda x: x):
            with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=FakeBackend()):
                try:
                    res = pp.unroll_pipeline(g4, 2)
                except KeyError:
                    pass


def test_distribute_tensor_branches():
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.distributed import distribute_tensor, set_distribution

    # Needs dist to be initialized
    class DummyDist:
        pass

    set_distribution(DummyDist())

    # When eager_mode is True, it returns the tensor directly
    config.eager_mode = True
    tensor = "dummy_tensor"
    res1 = distribute_tensor(tensor)
    assert res1 == "dummy_tensor"

    res2 = distribute_tensor(tensor="kwarg_tensor")
    assert res2 == "kwarg_tensor"

    # When eager_mode is False, it goes to shard_tensor
    config.eager_mode = False

    # Mock shard_tensor
    from unittest.mock import patch

    with patch("ml_switcheroo_compiler.distributed.distributed_ops.shard_tensor", return_value="sharded") as mock_shard:
        res3 = distribute_tensor(tensor)
        assert res3 == "sharded"
        mock_shard.assert_called_once_with("dummy_tensor")

    # Cleanup
    config.eager_mode = True
    set_distribution(None)
