from unittest.mock import patch

from ml_switcheroo_compiler.distributed.strategy import (
    CentralStorageStrategy,
    Coordinator,
    KubernetesClusterResolver,
    MeshShardingStrategy,
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


def test_strategy_classes():
    ps = ParameterServerStrategy(cluster_resolver=1)
    assert ps.cluster_resolver == 1

    mw = MultiWorkerMirroredStrategy(cluster_resolver=2)
    assert mw.cluster_resolver == 2

    cs = CentralStorageStrategy()
    assert isinstance(cs, CentralStorageStrategy)

    tpu = TPUStrategy(tpu_cluster_resolver=3)
    assert tpu.tpu_cluster_resolver == 3

    pch = PreemptionCheckpointHandler(cluster_resolver=4, checkpoint_dir="test")
    assert pch.cluster_resolver == 4
    assert pch.checkpoint_dir == "test"

    coord = Coordinator()
    assert not coord.joined
    coord.join()
    assert coord.joined

    pw = PerWorkerValue([1, 2])
    assert pw.values == [1, 2]

    rv = RemoteValue()
    assert rv.value is None


def test_server_methods():
    srv = Server(server_def=None)

    class DummyBackendNoSupport:
        __name__ = "dummy"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=DummyBackendNoSupport()):
        srv.start()
        srv.join()
        assert hasattr(srv, "_running")

    class DummyBackendSupport:
        __name__ = "dummy"

        def __init__(self):
            self.started = False
            self.joined = False

        def start_server(self, server):
            self.started = True

        def join_server(self, server):
            self.joined = True

    dummy = DummyBackendSupport()
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=dummy):
        srv.start()
        assert dummy.started
        srv.join()
        assert dummy.joined


def test_cluster_resolvers():
    import os

    with patch.dict(os.environ, {}, clear=True):
        slurm = SlurmClusterResolver()
        assert slurm.cluster == {}

        kube = KubernetesClusterResolver()
        assert kube.cluster == {"worker": ["localhost:8080"]}

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "host1,host2"}):
        slurm = SlurmClusterResolver()
        assert slurm.cluster == {"worker": ["host1", "host2"]}

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1", "HOSTNAME": "pod1"}):
        kube = KubernetesClusterResolver()
        assert kube.cluster == {"worker": ["pod1:8080"]}


def test_tf_config_valid():
    import json
    import os

    with patch.dict(os.environ, {"TF_CONFIG": json.dumps({"cluster": {"worker": ["host1", "host2"]}})}):
        resolver = TFConfigClusterResolver()
        assert "worker" in resolver.cluster


def test_mesh_sharding_strategy():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    class MockLayoutMap:
        def get(self, id):
            if id == "n3":
                return "spec3"
            return None

    strat = MeshShardingStrategy(mesh=None, layout_map=MockLayoutMap())

    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Constant")
    n1.sharding = "spec1"

    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])
    n3 = LogicalNode(id="n3", op_type="Sub", inputs=[])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    with patch("ml_switcheroo_compiler.transforms.passes.spmd.inject_spmd_communication_pass", return_value=True):
        res = strat.lower_sharding(g)
        assert res is True

    assert g.nodes["n1"].sharding == "spec1"  # unchanged
    assert g.nodes["n2"].sharding == "spec1"  # propagated from n1
    assert g.nodes["n3"].sharding == "spec3"  # from layout_map


def test_tf_config_invalid():
    import os

    from ml_switcheroo_compiler.distributed.strategy import TFConfigClusterResolver

    with patch.dict(os.environ, {"TF_CONFIG": "invalid json"}):
        resolver = TFConfigClusterResolver()
        assert resolver.cluster == {}


def test_mesh_sharding_strategy_no_layout():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy

    strat = MeshShardingStrategy()

    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Constant")
    n1.sharding = "spec1"

    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1", "missing"])
    n3 = LogicalNode(id="n3", op_type="Sub", inputs=["n2"])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2
    g.nodes["n3"] = n3

    strat.propagate_layouts(g)

    assert g.nodes["n1"].sharding == "spec1"
    assert g.nodes["n2"].sharding == "spec1"
    assert g.nodes["n3"].sharding == "spec1"


def test_tf_config_empty():
    import os

    from ml_switcheroo_compiler.distributed.strategy import TFConfigClusterResolver

    with patch.dict(os.environ, clear=True):
        resolver = TFConfigClusterResolver()
        assert resolver.cluster == {}


def test_mesh_sharding_strategy_all_inputs_no_sharding():
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy

    strat = MeshShardingStrategy()

    g = LogicalGraph()
    n1 = LogicalNode(id="n1", op_type="Constant")

    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    strat.propagate_layouts(g)

    assert getattr(g.nodes["n1"], "sharding", None) is None
    assert getattr(g.nodes["n2"], "sharding", None) is None
