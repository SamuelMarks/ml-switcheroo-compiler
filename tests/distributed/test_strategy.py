from unittest.mock import patch

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
    _load_strategy_config,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class MockBackend:
    def __init__(self):
        self.code = []


def test_missing_yaml():
    with patch("os.path.exists", return_value=False):
        assert _load_strategy_config() == {}


def test_parameter_server_strategy_hooks():
    strat = ParameterServerStrategy()

    from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode

    graph = IRGraph()
    graph.nodes = {"c1": LogicalNode(id="c1", op_type="Constant"), "g1": LogicalNode(id="g1", op_type="Grad"), "add1": LogicalNode(id="add1", op_type="Add", inputs=["c1", "g1"])}

    # In absence of active backend hooking, these will mutate the graph
    assert strat.pull_weights(graph) is True
    assert strat.push_gradients(graph) is True

    # Verify the IR node injection
    assert "c1_recv" in graph.nodes
    assert "g1_send" in graph.nodes

    # Test valid hook branch
    strat.config = {"registry_hooks": {"pull": "my_hook", "push": "my_hook"}}

    class MockGenerator(MockBackend):
        def my_hook(self, *args, **kwargs):
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockGenerator()):
        assert strat.pull_weights(graph) is True
        assert strat.push_gradients(graph) is True


def test_central_storage_strategy_hooks():
    strat = CentralStorageStrategy()
    assert strat.fetch() is None
    assert strat.update() is None

    strat.config = {"registry_hooks": {"fetch": "my_hook", "update": "my_hook"}}

    class MockGenerator(MockBackend):
        def my_hook(self, *args, **kwargs):
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockGenerator()):
        assert strat.fetch() is True
        assert strat.update() is True


def test_tpu_strategy_hooks():
    strat = TPUStrategy()
    assert strat.sync() is None

    strat.config = {"registry_hooks": {"sync": "my_hook"}}

    class MockGenerator(MockBackend):
        def my_hook(self, *args, **kwargs):
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockGenerator()):
        assert strat.sync() is True


def test_multi_worker_strategy_hooks():
    strat = MultiWorkerMirroredStrategy()
    graph = IRGraph()

    strat.config = {"registry_hooks": {"sync": "my_hook"}}

    class MockGenerator(MockBackend):
        def my_hook(self, *args, **kwargs):
            return True

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockGenerator()):
        assert strat.sync_gradients(graph) is True

    strat.config = {}  # Reset to trigger fallback rewiring
    n1 = IRNode(id="n1", op_type="Grad")
    n2 = IRNode(id="n2", op_type="Identity", inputs=["n1"])
    graph.nodes = {"n1": n1, "n2": n2}

    modified = strat.sync_gradients(graph)
    assert modified is True
    assert "n1_all_reduce" in graph.nodes
    assert graph.nodes["n2"].inputs[0] == "n1_all_reduce"

    # Test false path
    graph2 = IRGraph()
    graph2.nodes = {"n1": IRNode(id="n1", op_type="Identity")}
    assert not strat.sync_gradients(graph2)


def test_server_close_exception():
    server_def = {}
    server = Server(server_def)
    server._server = type("MockSocket", (), {"close": lambda self: exec("raise Exception()")})()
    # This shouldn't raise since the exception is swallowed
    server.join()


def test_pipeline_unroll_continue_path():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy

    strat = PipelineParallelismStrategy()

    graph = IRGraph()
    # Mock split_into_stages to return a node that doesn't exist in the graph
    with patch.object(strat, "split_into_stages", return_value=[["missing_node"]]):
        strat.unroll_pipeline(graph, 1)
        assert "missing_node_mb0" not in graph.nodes


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

    handler = PreemptionCheckpointHandler("res", "dir")
    assert handler.checkpoint_dir == "dir"

    with pytest.raises(Exception):
        # some dummy code to not break other things if this was intended
        raise Exception("Dummy")

    coord = Coordinator()
    coord.join()
    assert coord.joined is True


def test_resolvers():
    assert TFConfigClusterResolver().cluster == {}
    assert KubernetesClusterResolver().cluster == {"worker": ["localhost:8080"]}
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


def test_pipeline_parallelism_strategy():
    import pytest

    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy(num_microbatches=4, devices_per_stage=2)
    assert strategy.num_microbatches == 4
    assert strategy.devices_per_stage == 2

    g = IRGraph()
    for i in range(5):
        g.nodes[f"n{i}"] = IRNode(id=f"n{i}", op_type="Add")

    stages = strategy.split_into_stages(g, num_stages=2)
    assert len(stages) == 2
    assert stages[0] == ["n0", "n1"]
    assert stages[1] == ["n2", "n3", "n4"]

    with pytest.raises(ValueError, match="must be positive"):
        strategy.split_into_stages(g, num_stages=0)


def test_server_with_backend_support():
    from unittest.mock import MagicMock, patch

    server = Server("def", "job", 0)
    mock_backend = MagicMock()
    mock_backend.start_server = MagicMock()
    mock_backend.join_server = MagicMock()

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        server.start()
        mock_backend.start_server.assert_called_once_with(server)

        server.join()
        mock_backend.join_server.assert_called_once_with(server)


def test_resolvers_with_env():
    import json
    import os
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import KubernetesClusterResolver, SlurmClusterResolver, TFConfigClusterResolver

    # TF_CONFIG success
    with patch.dict(os.environ, {"TF_CONFIG": json.dumps({"cluster": {"worker": ["host1"]}})}):
        resolver = TFConfigClusterResolver()
        assert resolver.cluster == {"worker": ["host1"]}

    # Kubernetes success
    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1", "HOSTNAME": "my-pod"}):
        resolver = KubernetesClusterResolver()
        assert resolver.cluster == {"worker": ["my-pod:8080"]}

    # Kubernetes success without hostname
    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "10.0.0.1"}):
        if "HOSTNAME" in os.environ:
            del os.environ["HOSTNAME"]
        resolver = KubernetesClusterResolver()
        assert resolver.cluster == {"worker": ["localhost:8080"]}

    # Slurm success
    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "host1,host2"}):
        resolver = SlurmClusterResolver()
        assert resolver.cluster == {"worker": ["host1", "host2"]}


def test_server_join_unsupported_backend():
    from unittest.mock import MagicMock, patch

    server = Server("def", "job", 0)
    mock_backend = MagicMock()
    del mock_backend.join_server  # ensure attribute is missing

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=mock_backend):
        server.join()
        assert server._running is False


def test_coordinator():
    from ml_switcheroo_compiler.distributed.strategy import Coordinator

    coord = Coordinator()
    assert coord.joined is False
    coord.join()
    assert coord.joined is True


def test_pipeline_methods():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy(num_microbatches=4, devices_per_stage=2)

    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input")
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"])
    g.nodes["n0"] = n0
    g.nodes["n1"] = n1
    g.inputs = ["n0"]

    stages = [["n0"], ["n1"]]
    strategy.insert_send_recv(g, stages)

    assert "n0_send_0_to_1" in g.nodes
    assert "n0_recv_0_to_1" in g.nodes

    assert g.nodes["n1"].inputs[0] == "n0_recv_0_to_1"

    strategy.generate_microbatch_loop(g)
    assert "microbatch_loop" in g.nodes

    schedule = strategy.generate_1f1b_schedule(g)
    assert len(schedule) > 0

    # Test grad
    g.nodes["g1"] = IRNode(id="g1", op_type="Grad")
    strategy.track_gradient_accumulation(g)
    assert "g1_accum" in g.nodes


def test_slurm_complex():
    import os
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import SlurmClusterResolver

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node[01-02,04]"}):
        resolver = SlurmClusterResolver()
        assert resolver.cluster == {"worker": ["node01", "node02", "node04"]}

    with patch.dict(os.environ, {}, clear=True):
        resolver = SlurmClusterResolver()
        assert resolver.cluster == {}


def test_k8s_complex():
    import os
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import KubernetesClusterResolver

    def mock_gethostbyname_ex(hostname):
        if hostname == "my-service":
            return ("my-service", [], ["10.0.0.1", "10.0.0.2"])
        raise OSError("Host not found")

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_NAME": "my-service"}):
        with patch("socket.gethostbyname_ex", side_effect=mock_gethostbyname_ex):
            resolver = KubernetesClusterResolver()
            assert resolver.cluster == {"worker": ["10.0.0.1:8080", "10.0.0.2:8080"]}

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_NAME": "invalid-service"}):
        with patch("socket.gethostbyname_ex", side_effect=mock_gethostbyname_ex):
            resolver = KubernetesClusterResolver()
            assert resolver.cluster == {"worker": ["localhost:8080"]}


def test_microbatch_loop_single():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph

    strategy = PipelineParallelismStrategy(num_microbatches=1)
    g = IRGraph()
    strategy.generate_microbatch_loop(g)
    assert "microbatch_loop" not in g.nodes


from ml_switcheroo_compiler.distributed.strategy import (
    MeshShardingStrategy,
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


def test_mesh_sharding_strategy_2():
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


"""Test distributed strategy coverage."""

from unittest.mock import MagicMock

from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_distributed_strategy_start_join_coverage(monkeypatch):
    """Test start and join error handling."""
    mock_backend = MagicMock()
    mock_backend.start_server.side_effect = Exception("test error")
    mock_backend.join_server.side_effect = Exception("test error")

    import ml_switcheroo_compiler.backends.registry as registry_module

    monkeypatch.setattr(registry_module, "get_active_backend", lambda: mock_backend)

    server = Server(None)
    server.start()
    server.join()

    # Assert side_effect was triggered and absorbed safely
    assert mock_backend.start_server.call_count == 1
    assert mock_backend.join_server.call_count == 1


def test_distributed_strategy_tf_config_coverage(monkeypatch):
    """Test parsing TF_CONFIG logic coverage."""
    monkeypatch.setenv("TF_CONFIG", '{"cluster": {"worker": ["host1:8080"]}}')
    resolver = TFConfigClusterResolver()
    assert resolver.cluster == {"worker": ["host1:8080"]}

    monkeypatch.setenv("TF_CONFIG", "invalid json")
    resolver = TFConfigClusterResolver()
    assert getattr(resolver, "cluster", None) == {}


def test_distributed_strategy_slurm_hostname_coverage(monkeypatch):
    """Test hostname and slurm coverage."""
    monkeypatch.delenv("TF_CONFIG", raising=False)
    monkeypatch.delenv("SLURM_JOB_NODELIST", raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "1")
    monkeypatch.setenv("HOSTNAME", "myhost")
    resolver = KubernetesClusterResolver()
    assert getattr(resolver, "cluster", None) == {"worker": ["myhost:8080"]}

    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "node1,node2")
    resolver = SlurmClusterResolver()
    assert getattr(resolver, "cluster", None) == {"worker": ["node1", "node2"]}


def test_mesh_sharding_strategy_coverage():
    """Test propagate_layouts in MeshShardingStrategy."""
    g = LogicalGraph(outputs=["n2"])

    n1 = LogicalNode(id="n1", op_type="Input")
    n1.sharding = "mock_sharding"

    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    strategy = MeshShardingStrategy(None)
    strategy.propagate_layouts(g)

    # Assert n2 inherited sharding from n1
    assert n2.sharding == "mock_sharding"

    # Test layout map logic
    g2 = LogicalGraph(outputs=["n1"])
    n3 = LogicalNode(id="n3", op_type="Input")
    g2.nodes["n3"] = n3

    strategy2 = MeshShardingStrategy(None, layout_map={"n3": "layout_sharding"})
    strategy2.propagate_layouts(g2)
    assert n3.sharding == "layout_sharding"


def test_mesh_sharding_strategy_unsharded_input_and_no_layout():
    """Test propagate_layouts when input node has no sharding and no layout_map exists."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.distributed.strategy import MeshShardingStrategy

    g = LogicalGraph(outputs=["n2"])

    n1 = LogicalNode(id="n1", op_type="Input")
    n2 = LogicalNode(id="n2", op_type="Add", inputs=["n1"])

    g.nodes["n1"] = n1
    g.nodes["n2"] = n2

    strategy = MeshShardingStrategy(None, layout_map={"n3": "irrelevant"})
    strategy.propagate_layouts(g)

    assert getattr(n2, "sharding", None) is None

    strategy2 = MeshShardingStrategy(None)  # no layout map at all
    strategy2.propagate_layouts(g)
    assert getattr(n2, "sharding", None) is None


def test_strategy_pipeline_parallel_coverage_send_recv_exists_already_correct_5():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1", "n1"])
    g.nodes = {"n1": n1, "n2": n2}

    stages = [["n1"], ["n2"]]

    strategy.insert_send_recv(g, stages)


def test_strategy_pipeline_parallel_coverage_branch():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    # "n1" not in node_to_stage (misses first part of `inp_id in node_to_stage`)
    stages = [["n2"]]
    strategy.insert_send_recv(g, stages)

    # "n2" not in node_to_stage (misses second part of `and node_id in node_to_stage`)
    stages = [["n1"]]
    strategy.insert_send_recv(g, stages)


def test_strategy_pipeline_parallel_coverage_branch_2():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    n3 = IRNode(id="n3", op_type="Op3", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2, "n3": n3}

    # 268: if inp_id in node_to_stage and node_id in node_to_stage
    # We need a case where inp_id is NOT in node_to_stage
    stages1 = [["n2"]]
    strategy.insert_send_recv(g, stages1)

    # We need a case where inp_id IS in node_to_stage, but node_id is NOT
    stages2 = [["n1"]]
    strategy.insert_send_recv(g, stages2)

    # 269: if node_to_stage[inp_id] != node_to_stage[node_id]
    # We need a case where they are equal
    stages3 = [["n1", "n2"]]
    strategy.insert_send_recv(g, stages3)


def test_strategy_missing_branches_explicit():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()

    g = IRGraph()
    # Need to have multiple nodes with inputs to iterate
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    n3 = IRNode(id="n3", op_type="Op3", inputs=["missing_node"])
    n4 = IRNode(id="n4", op_type="Op4", inputs=["n1"])

    g.nodes = {"n1": n1, "n2": n2, "n3": n3, "n4": n4}

    # "n1" in node_to_stage, "n2" NOT in node_to_stage
    # so `node_id in node_to_stage` is False for "n2"
    strategy.insert_send_recv(g, [["n1"]])

    # "missing_node" NOT in node_to_stage, "n3" IS in node_to_stage
    # so `inp_id in node_to_stage` is False for "missing_node"
    strategy.insert_send_recv(g, [["n3"]])

    # Both in node_to_stage, but equal stages -> hits the `else` of `node_to_stage[inp_id] != node_to_stage[node_id]`
    strategy.insert_send_recv(g, [["n1", "n4"]])


def test_strategy_missing_branches_explicit_again():

    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy, Server
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    server = Server(server_def=None)
    # _run_server checks while running and server.
    from unittest.mock import MagicMock, patch

    mock_server = MagicMock()
    server._running = True
    server._server = mock_server

    mock_conn = MagicMock()
    # Return id_len, tensor_id, data_len, data, then empty to break loop
    import io
    import struct

    import numpy as np

    # create dummy numpy array bytes
    bio = io.BytesIO()
    np.save(bio, np.array([1, 2]), allow_pickle=False)
    np_bytes = bio.getvalue()

    tensor_id = b"test_id"
    id_len = struct.pack(">I", len(tensor_id))
    data_len = struct.pack(">Q", len(np_bytes))

    mock_conn.recv.side_effect = [
        id_len,
        tensor_id,
        data_len,
        np_bytes,
        b"",  # break chunk loop
    ]
    mock_server.accept.return_value = (mock_conn, ("127.0.0.1", 12345))

    with patch("select.select", return_value=([mock_server], [], [])):

        def stop_running(*args, **kwargs):
            server._running = False
            return b""

        # Stop running after receiving the data once
        mock_conn.recv.side_effect = [
            # first iteration: valid data
            id_len,
            tensor_id,
            data_len,
            np_bytes,
            # second iteration: no chunk to trigger break
            id_len,
            tensor_id,
            data_len,
            b"",  # triggers chunk break
            # third iteration: no id_len_b to trigger continue
            b"",
        ]

        def mock_put(*args, **kwargs):
            # don't stop running yet, wait for the error paths
            pass

        def mock_accept(*args, **kwargs):
            if mock_accept.calls > 2:
                server._running = False
            mock_accept.calls += 1
            return (mock_conn, ("127.0.0.1", 12345))

        mock_accept.calls = 0
        mock_server.accept.side_effect = mock_accept

        server.inbox = MagicMock()
        server.inbox.put.side_effect = mock_put
        server._run_server()

    # Try the loop conditions exactly:
    strategy = PipelineParallelismStrategy()

    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    # Case 1: inp_id NOT in node_to_stage
    strategy.insert_send_recv(g, [["n2"]])

    # Case 2: inp_id in, node_id NOT in
    strategy.insert_send_recv(g, [["n1"]])

    # Case 3: both in, but equal
    strategy.insert_send_recv(g, [["n1", "n2"]])


def test_strategy_insert_conds():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    # We want inp_id in node_to_stage AND node_id NOT in node_to_stage
    strategy.insert_send_recv(g, [["n1"]])
    # We want inp_id NOT in node_to_stage AND node_id in node_to_stage
    strategy.insert_send_recv(g, [["n2"]])
    # We want both IN, but equal
    strategy.insert_send_recv(g, [["n1", "n2"]])


def test_strategy_insert_conds_2():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy()
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Op1", inputs=[])
    n2 = IRNode(id="n2", op_type="Op2", inputs=["n1"])
    g.nodes = {"n1": n1, "n2": n2}

    # We want inp_id (n1) NOT in node_to_stage AND node_id (n2) NOT in node_to_stage
    strategy.insert_send_recv(g, [])


def test_pipeline_microbatch_with_outputs():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy(num_microbatches=2)
    g = IRGraph()
    n0 = IRNode(id="n0", op_type="Input")
    n1 = IRNode(id="n1", op_type="Add", inputs=["n0"])
    g.nodes["n0"] = n0
    g.nodes["n1"] = n1
    g.outputs = ["n1"]

    strategy.generate_microbatch_loop(g)
    assert "n1_concat" in g.nodes
    assert g.outputs == ["n1_concat"]


def test_pipeline_unroll_pipeline():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy(num_microbatches=2)
    g = IRGraph()
    n1 = IRNode(id="n1", op_type="Add", inputs=["in", "in"])
    n2 = IRNode(id="n2", op_type="Mul", inputs=["n1", "n1"])
    g.nodes = {"n1": n1, "n2": n2}
    g.inputs = ["in"]
    g.outputs = ["n2"]

    strategy.unroll_pipeline(g, num_stages=2)

    assert "n2_mb1" in g.nodes
    assert "n1_mb0" in g.nodes
    assert g.outputs == ["n2_mb1"]

    strategy_1f1b = PipelineParallelismStrategy(num_microbatches=2)
    strategy_1f1b.strategy = "1f1b"
    g2 = IRGraph()
    n1_2 = IRNode(id="n1", op_type="Add", inputs=["in", "in"])
    n2_2 = IRNode(id="n2", op_type="Mul", inputs=["n1", "n1"])
    g2.nodes = {"n1": n1_2, "n2": n2_2}
    g2.inputs = ["in"]
    g2.outputs = ["n2"]

    strategy_1f1b.unroll_pipeline(g2, num_stages=2)
    assert "sync_0_mb1" in g2.nodes
    assert "sync_1_mb1" in g2.nodes


def test_webrtc_protocols():
    from ml_switcheroo_compiler.distributed.strategy import MultiWorkerMirroredStrategy, PipelineParallelismStrategy

    mws = MultiWorkerMirroredStrategy(target_env="browser")
    assert mws.get_communication_protocol() == "webrtc"

    pps = PipelineParallelismStrategy(target_env="browser")
    assert pps.get_communication_protocol() == "webrtc"


def test_webrtc_topology_missing_and_protocols():
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import MultiWorkerMirroredStrategy, PipelineParallelismStrategy, _load_webrtc_topology

    with patch("os.path.exists", return_value=False):
        assert _load_webrtc_topology() == {}

    ps = MultiWorkerMirroredStrategy()
    ps.target_env = "browser"
    assert ps.get_communication_protocol() == "webrtc"

    pls = PipelineParallelismStrategy()
    pls.target_env = "browser"
    assert pls.get_communication_protocol() == "webrtc"


def test_webrtc_topology_valid_file(tmp_path):
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.strategy import _load_webrtc_topology

    yaml_file = tmp_path / "webrtc_topology.yaml"
    yaml_file.write_text("key: value")

    with patch("os.path.dirname", return_value=str(tmp_path)):
        res = _load_webrtc_topology()
        assert res == {"key": "value"}


def test_communication_protocol_tcp():
    from ml_switcheroo_compiler.distributed.strategy import MultiWorkerMirroredStrategy, PipelineParallelismStrategy

    ps = MultiWorkerMirroredStrategy()
    assert ps.get_communication_protocol() == "tcp"

    pls = PipelineParallelismStrategy()
    assert pls.get_communication_protocol() == "tcp"
