from ml_switcheroo_compiler.distributed.strategy import (
    CentralStorageStrategy,
    Coordinator,
    KubernetesClusterResolver,
    MultiWorkerMirroredStrategy,
    ParameterServerStrategy,
    PerWorkerValue,
    PreemptionCheckpointHandler,
    RemoteValue,
    Server,
    SlurmClusterResolver,
    TFConfigClusterResolver,
    TPUStrategy,
)


def test_strategies():
    pss = ParameterServerStrategy(cluster_resolver="resolver")
    assert pss.cluster_resolver == "resolver"

    mws = MultiWorkerMirroredStrategy(cluster_resolver="resolver")
    assert mws.cluster_resolver == "resolver"

    css = CentralStorageStrategy()
    assert css is not None

    tpus = TPUStrategy(tpu_cluster_resolver="tpu_resolver")
    assert tpus.tpu_cluster_resolver == "tpu_resolver"


def test_handlers_and_servers():
    import pytest

    with pytest.raises(Exception):
        handler = PreemptionCheckpointHandler("res", "dir")
        assert handler.checkpoint_dir == "dir"

        server = Server("def", "job", 0)
        server.start()
        server.join()
        assert server.job_name == "job"

        coord = Coordinator()
        coord.join()
        assert coord.joined is True


def test_resolvers():
    assert TFConfigClusterResolver().cluster == {}
    assert KubernetesClusterResolver().cluster == {}
    assert SlurmClusterResolver().cluster == {}


def test_values():
    pv = PerWorkerValue([1, 2])
    assert pv.values == [1, 2]

    rv = RemoteValue()
    assert rv.value is None


def test_mesh_sharding_strategy():
    from ml_switcheroo_compiler.distributed.layout_map import LayoutMap, ShardingSpec
    from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    class DummySharding:
        def __init__(self, mapping):
            self.mesh_mapping = mapping

    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], sharding=DummySharding(["x"]))
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"])

    g.nodes["n0"] = n0
    g.nodes["n1"] = n1

    strategy = MeshShardingStrategy()
    strategy.propagate_layouts(g)

    # Sharding should have propagated from n0 to n1
    assert n1.sharding is not None
    assert n1.sharding.mesh_mapping == ["x"]

    # Test with layout map
    g2 = IRGraph()
    n2 = IRNode(id="n2", op_type="Input", inputs=[])
    g2.nodes["n2"] = n2

    layout_map = LayoutMap()
    spec = ShardingSpec("mesh", ["y"])
    layout_map.insert("n2", spec)

    strategy2 = MeshShardingStrategy(layout_map=layout_map)
    strategy2.propagate_layouts(g2)
    assert n2.sharding == spec

    # Test lower sharding
    g3 = IRGraph()
    n3_in = IRNode(id="in1", op_type="Input", inputs=[], sharding=DummySharding(["x"]))
    n3_out = IRNode(id="n3", op_type="Add", inputs=["in1"], sharding=DummySharding([None]))
    g3.nodes["in1"] = n3_in
    g3.nodes["n3"] = n3_out

    strategy3 = MeshShardingStrategy()
    modified = strategy3.lower_sharding(g3)
    assert modified is True
    # The communication pass should have injected an all_gather node
    assert "in1_all_gather" in g3.nodes


def test_exchange_ipc_data_failure():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data

    arr = np.array([1.0, 2.0])
    # Connect to invalid port / authkey or coordinate fails, should fallback and return [arr]*size
    res = _exchange_ipc_data(rank=1, size=2, tensor_data=arr)
    assert len(res) == 2
    assert np.allclose(res[0], arr)
    assert np.allclose(res[1], arr)
