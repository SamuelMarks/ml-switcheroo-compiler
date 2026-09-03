import io
import json
import os
import socket
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from ml_switcheroo_compiler.distributed.config_models import MeshMappingConfig, MicrobatchSplittingConfig, ScheduleConfig, SchedulePhaseConfig, StageCommunicationConfig, TopologyConfig
from ml_switcheroo_compiler.distributed.strategy import (
    CentralStorageStrategy,
    Coordinator,
    KubernetesClusterResolver,
    MeshShardingStrategy,
    MultiWorkerMirroredStrategy,
    ParameterServerStrategy,
    PerWorkerValue,
    PipelineParallelismStrategy,
    PreemptionCheckpointHandler,
    RemoteValue,
    Server,
    SlurmClusterResolver,
    TFConfigClusterResolver,
    TPUStrategy,
    _load_webrtc_topology,
)
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _make_graph():
    graph = IRGraph()
    for i in range(5):
        node = IRNode(id=f"node_{i}", op_type="Add", inputs=[f"node_{i - 1}"] if i > 0 else [], attributes={})
        graph.nodes[f"node_{i}"] = node

    grad_node = IRNode(id="grad_node", op_type="Grad", inputs=["node_4"], attributes={})
    graph.nodes["grad_node"] = grad_node
    graph.outputs = ["grad_node"]
    return graph


def test_parameter_server_strategy():
    strategy = ParameterServerStrategy()
    graph = _make_graph()
    graph.nodes["const_node"] = IRNode(id="const_node", op_type="Constant", inputs=[], attributes={})
    graph.nodes["node_1"].inputs.append("const_node")
    graph.nodes["add_after"] = IRNode(id="add_after", op_type="Add", inputs=["grad_node"], attributes={})
    strategy.push_gradients(graph)
    strategy.pull_weights(graph)
    assert not strategy.push_gradients(IRGraph())


def test_central_storage_strategy():
    strategy = CentralStorageStrategy()
    assert strategy.fetch() is None
    assert strategy.update() is None


def test_multi_worker_mirrored():
    strategy = MultiWorkerMirroredStrategy()
    graph = _make_graph()
    assert strategy.sync_gradients(graph)
    assert any(n.op_type == "AllReduce" for n in graph.nodes.values())

    # Try with empty graph
    assert not strategy.sync_gradients(IRGraph())
    assert isinstance(strategy.get_communication_protocol(), str)
    s2 = MultiWorkerMirroredStrategy(target_env="browser")
    assert s2.get_communication_protocol() == "webrtc"


def test_tpu_strategy():
    strategy = TPUStrategy()
    with pytest.raises(RuntimeError, match="TPU sync is only supported"):
        strategy.sync()


def test_pipeline_parallel_strategy():
    strategy = PipelineParallelismStrategy(num_microbatches=4, topology_name="invalid_one")
    # Will use fallback config
    graph = _make_graph()
    assert strategy.lower(graph)
    assert "pipeline_schedule" in graph.attributes
    assert not strategy.lower(IRGraph())

    strategy_browser = PipelineParallelismStrategy(target_env="browser")
    assert strategy_browser.get_communication_protocol() == "webrtc"

    with patch("ml_switcheroo_compiler.distributed.strategy._load_strategy_config", return_value={}):
        strategy_noyaml = PipelineParallelismStrategy(num_microbatches=4)
        strategy_noyaml.lower(_make_graph())

    # Test yaml driven schedule by mocking _load_strategy_config
    config = TopologyConfig(
        microbatch_splitting=MicrobatchSplittingConfig(num_microbatches=4, strategy="1f1b"),
        mesh_mapping=MeshMappingConfig(devices_per_stage=1),
        stage_communication=StageCommunicationConfig(protocol="grpc"),
        schedule=ScheduleConfig(
            phases=[
                SchedulePhaseConfig(type="warmup", count_expression="num_stages", operations=["forward"]),
                SchedulePhaseConfig(type="steady", count_expression="num_microbatches", operations=["forward", "backward"]),
                SchedulePhaseConfig(type="cooldown", count_expression="num_stages", operations=["backward"]),
            ]
        ),
    )
    with patch("ml_switcheroo_compiler.distributed.strategy._load_strategy_config") as m:
        m.return_value = {"custom": config}
        strategy = PipelineParallelismStrategy(topology_name="custom")
        graph = _make_graph()
        assert strategy.lower(graph)
        assert isinstance(strategy.get_communication_protocol(), str)
    s2 = MultiWorkerMirroredStrategy(target_env="browser")
    assert s2.get_communication_protocol() == "webrtc"

    with pytest.raises(ValueError):
        strategy.split_into_stages(graph, 0)

    with patch("ml_switcheroo_compiler.distributed.config_models.PipelineTopologiesConfig.get", return_value=None):
        with pytest.raises(ValueError):
            PipelineParallelismStrategy(topology_name="not_found")


def test_mesh_sharding_strategy():
    strategy = MeshShardingStrategy()
    graph = _make_graph()
    # propagate_layouts is pass
    strategy.propagate_layouts(graph)

    graph.nodes["node_1"].sharding = "shard1"
    strategy.lower_sharding(graph)
    strategy.lower_sharding(IRGraph())


def test_resolvers():
    with patch.dict(os.environ, {"TF_CONFIG": json.dumps({"cluster": {"worker": ["host1:80", "host2:80"]}})}):
        res = TFConfigClusterResolver()
        assert len(res.cluster["worker"]) == 2

    with patch.dict(os.environ, {"TF_CONFIG": "invalid json"}):
        res = TFConfigClusterResolver()
        assert not res.cluster

    with patch.dict(os.environ, {"MASTER_ADDR": "host1", "MASTER_PORT": "80", "KUBERNETES_SERVICE_NAME": "svc"}):
        with patch("socket.gethostbyname_ex", return_value=(None, None, ["1.1.1.1"])):
            res = KubernetesClusterResolver()
            assert res.cluster["worker"] == ["1.1.1.1:80"]

        with patch("socket.gethostbyname_ex", side_effect=OSError):
            res = KubernetesClusterResolver()
            assert res.cluster["worker"] == ["host1:80"]

    with patch.dict(os.environ, {"MASTER_ADDR": "host1", "MASTER_PORT": "80", "KUBERNETES_SERVICE_HOST": "host2"}):
        if "KUBERNETES_SERVICE_NAME" in os.environ:
            del os.environ["KUBERNETES_SERVICE_NAME"]
        res = KubernetesClusterResolver()
        assert res.cluster["worker"] == ["host1:80"]  # HOSTNAME is not set

    with patch.dict(os.environ, {"MASTER_ADDR": "host1", "MASTER_PORT": "80"}):
        if "KUBERNETES_SERVICE_NAME" in os.environ:
            del os.environ["KUBERNETES_SERVICE_NAME"]
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            del os.environ["KUBERNETES_SERVICE_HOST"]
        res = KubernetesClusterResolver()
        assert res.cluster["worker"] == ["host1:80"]

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node[01-03]"}):
        res = SlurmClusterResolver()
        assert len(res.cluster["worker"]) == 3

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node1"}):
        res = SlurmClusterResolver()
        assert res.cluster["worker"] == ["node1"]

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node1,node2"}):
        res = SlurmClusterResolver()
        assert res.cluster["worker"] == ["node1", "node2"]

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": ""}):
        res = SlurmClusterResolver()
        assert not res.cluster


def test_server_and_coordinator():
    coord = Coordinator()
    assert not coord.joined
    coord.join()
    assert coord.joined

    server = Server(server_def={"hello": "world"})
    # Start and stop to cover thread
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = MagicMock()
        del mock_backend.return_value.start_server
        del mock_backend.return_value.join_server

        server.start()
        assert server._running

        # Connect to server to test loop
        addr = server._server.getsockname()
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(addr)

        # Test push
        header = json.dumps({"action": "push", "tensor_id": "t1"}).encode()

        bio = io.BytesIO()
        np.save(bio, np.array([1, 2, 3]), allow_pickle=False)
        data = bio.getvalue()

        payload = len(header).to_bytes(4, "big") + header + len(data).to_bytes(8, "big") + data
        # Test push twice to cover state_store update
        import time

        client.sendall(payload)

        client3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client3.connect(addr)
        client3.sendall(payload)
        time.sleep(0.1)

        # Test pull existing
        client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client2.connect(addr)
        header2 = json.dumps({"action": "pull", "tensor_id": "t1"}).encode()
        client2.sendall(len(header2).to_bytes(4, "big") + header2)

        # Test pull missing key
        client4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client4.connect(addr)
        header4 = json.dumps({"action": "pull", "tensor_id": "t_missing"}).encode()
        client4.sendall(len(header4).to_bytes(4, "big") + header4)

        time.sleep(0.1)

        server.join()
        assert not server._running


def test_backend_hooks():
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        backend_mock = MagicMock()
        mock_backend.return_value = backend_mock

        # ParameterServerStrategy pull/push
        ps = ParameterServerStrategy()
        ps.config = {"registry_hooks": {"pull": "pull_hook", "push": "push_hook"}}
        backend_mock.pull_hook.return_value = True
        backend_mock.push_hook.return_value = True
        assert ps.pull_weights(IRGraph()) is True
        assert ps.push_gradients(IRGraph()) is True

        # CentralStorageStrategy fetch/update
        cs = CentralStorageStrategy()
        cs.config = {"registry_hooks": {"fetch": "fetch_hook", "update": "update_hook"}}
        backend_mock.fetch_hook.return_value = "f"
        backend_mock.update_hook.return_value = "u"
        assert cs.fetch() == "f"
        assert cs.update() == "u"

        # TPUStrategy sync
        ts = TPUStrategy()
        ts.config = {"registry_hooks": {"sync": "sync_hook"}}
        backend_mock.sync_hook.return_value = "s"
        assert ts.sync() == "s"

        # MultiWorkerMirroredStrategy sync
        mw = MultiWorkerMirroredStrategy()
        mw.config = {"registry_hooks": {"sync": "sync_hook2"}}
        backend_mock.sync_hook2.return_value = True
        assert mw.sync_gradients(IRGraph()) is True

        # Server custom start/join
        server = Server()
        backend_mock.start_server.return_value = None
        backend_mock.join_server.return_value = None
        server.start()
        server.join()
        assert not server._running
    # Test server custom backend errors
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = MagicMock()
        mock_backend.return_value.start_server.side_effect = Exception("start err")
        mock_backend.return_value.join_server.side_effect = Exception("join err")

        server2 = Server()
        server2.start()
        # Should fallback to socket server
        assert server2._running
        server2.join()

    # Test server socket close exception
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend") as mock_backend:
        mock_backend.return_value = MagicMock()
        del mock_backend.return_value.start_server
        del mock_backend.return_value.join_server
        server3 = Server()
        server3._server = MagicMock()
        server3._thread = MagicMock()
        server3._server.close.side_effect = Exception("close err")
        server3.join()


def test_values():
    val = PerWorkerValue([1, 2, 3])
    assert val.values == [1, 2, 3]

    rem = RemoteValue()
    assert rem.value is None


def test_webrtc_topology():
    with patch("os.path.exists", return_value=False):
        assert _load_webrtc_topology() == {}

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="{}")):
            assert _load_webrtc_topology() == {}

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="strategies:\n  my_strat:\n    algorithm: ring")):
            from ml_switcheroo_compiler.distributed.strategy import _load_strategy_config

            cfg = _load_strategy_config()
            assert "my_strat" in cfg


def test_preemption():
    h = PreemptionCheckpointHandler(None, ".")
    assert h.checkpoint_dir == "."


def test_server_early_return():
    server4 = Server()
    server4._run_server()


def test_slurm_empty_nodes():
    with patch.dict("os.environ", {"SLURM_JOB_NODELIST": "node1"}):
        pass
