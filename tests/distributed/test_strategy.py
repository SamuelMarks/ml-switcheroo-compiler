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
    graph.nodes["scalar_const"] = IRNode(id="scalar_const", op_type="Constant", inputs=[], attributes={"is_scalar": True})
    graph.nodes["node_1"].inputs.extend(["const_node", "scalar_const"])
    graph.nodes["add_after"] = IRNode(id="add_after", op_type="Add", inputs=["grad_node"], attributes={})
    strategy.push_gradients(graph)
    strategy.pull_weights(graph)

    # Verify scalar constant was NOT replaced by Recv
    assert "scalar_const" in graph.nodes
    assert "scalar_const_recv" not in graph.nodes
    # Verify parameter constant was replaced by Recv with correct attributes
    assert "const_node_recv" in graph.nodes
    assert graph.nodes["const_node_recv"].attributes["src_rank"] == 0

    assert not strategy.push_gradients(IRGraph())
    # Empty graph -> modified is False, covering branch 87->90
    assert not strategy.pull_weights(IRGraph())


def test_is_parameter_node_branches():
    """Verify all conditional branches in ParameterServerStrategy._is_parameter_node."""
    # is_parameter is False -> False
    assert not ParameterServerStrategy._is_parameter_node(IRNode("n1", "Constant", attributes={"is_parameter": False}))
    # is_scalar_constant is True -> False
    assert not ParameterServerStrategy._is_parameter_node(IRNode("n2", "Constant", attributes={"is_scalar_constant": True}))
    # op_type in ("Parameter", "Variable", "Weight") -> True
    assert ParameterServerStrategy._is_parameter_node(IRNode("n3", "Parameter"))
    assert ParameterServerStrategy._is_parameter_node(IRNode("n4", "Variable"))
    assert ParameterServerStrategy._is_parameter_node(IRNode("n5", "Weight"))
    # is_parameter is True -> True
    assert ParameterServerStrategy._is_parameter_node(IRNode("n6", "CustomOp", attributes={"is_parameter": True}))
    # trainable is True -> True
    assert ParameterServerStrategy._is_parameter_node(IRNode("n7", "CustomOp", attributes={"trainable": True}))
    # role in ("parameter", "weight", "bias") -> True
    assert ParameterServerStrategy._is_parameter_node(IRNode("n8", "CustomOp", attributes={"role": "bias"}))
    # 0-d Constant without is_parameter -> False
    node_0d = IRNode("n9", "Constant")
    node_0d.shape_metadata = ()
    assert not ParameterServerStrategy._is_parameter_node(node_0d)


def test_load_webrtc_topology_missing():
    """Verify _load_webrtc_topology and _load_device_mesh_config return empty dict when file does not exist."""
    from ml_switcheroo_compiler.distributed.strategy import _load_device_mesh_config

    with patch("os.path.exists", return_value=False):
        assert _load_webrtc_topology() == {}
        assert _load_device_mesh_config() == {}


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


def test_data_parallel_sync_gradients_with_consumer():
    from ml_switcheroo_compiler.distributed.strategy import MultiWorkerMirroredStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = MultiWorkerMirroredStrategy()
    graph = IRGraph()
    grad_node = IRNode(id="g1", op_type="Grad", inputs=["x"])
    add_node = IRNode(id="a1", op_type="Add", inputs=["g1", "y"])
    graph.nodes = {"g1": grad_node, "a1": add_node}

    res = strategy.sync_gradients(graph)
    assert res is True
    assert "g1_all_reduce" in graph.nodes
    assert graph.nodes["a1"].inputs == ["g1_all_reduce", "y"]


def test_parameter_server_pull_weights_non_matching_consumer():
    """Test pull_weights when graph has consumer node not consuming the weight."""
    from ml_switcheroo_compiler.distributed.strategy import ParameterServerStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strat = ParameterServerStrategy()
    graph = IRGraph()
    w = IRNode(id="w1", op_type="Constant", inputs=[])
    c = IRNode(id="c1", op_type="Add", inputs=["other_input"])
    graph.nodes = {"w1": w, "c1": c}
    res = strat.pull_weights(graph)
    assert res is True


def test_server_run_unknown_action_and_send_action():
    """Test server loop with unknown action and send action."""
    import io
    import json
    from unittest.mock import MagicMock, patch

    import numpy as np

    from ml_switcheroo_compiler.distributed.strategy import Server

    server = Server()

    # 1. Unknown action e.g. "ping" (branch 335->305)
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

        mock_conn = MagicMock()
        header = json.dumps({"action": "ping", "tensor_id": "test"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        def mock_recv_ping(size):
            if size == 4:
                server._running = False
                return header_len
            return header

        mock_conn.recv.side_effect = mock_recv_ping
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])
        server._run_server()

    # 2. Action "send" (branch 349->305)
    with patch("select.select") as mock_select:
        mock_sock = MagicMock()
        server._server = mock_sock
        server._running = True

        mock_conn = MagicMock()
        header = json.dumps({"action": "send", "tensor_id": "test_send"}).encode("utf-8")
        header_len = len(header).to_bytes(4, "big")

        data_arr = np.array([1.0, 2.0])
        bio = io.BytesIO()
        np.save(bio, data_arr, allow_pickle=False)
        data = bio.getvalue()
        data_len = len(data).to_bytes(8, "big")

        class RecvSeqSend:
            def __init__(self):
                self.calls = 0

            def __call__(self, size):
                self.calls += 1
                if self.calls == 1:
                    return header_len
                elif self.calls == 2:
                    return header
                elif self.calls == 3:
                    return data_len
                elif self.calls == 4:
                    server._running = False
                    return data
                return b""

        mock_conn.recv.side_effect = RecvSeqSend()
        mock_sock.accept.return_value = (mock_conn, "addr")
        mock_select.return_value = ([mock_sock], [], [])
        server._run_server()
        assert not server.inbox.empty()

    # 3. Not ready branch (branch 308->305)
    server = Server()
    mock_sock = MagicMock()
    server._server = mock_sock
    server._running = True

    def toggle_not_ready(*args, **kwargs):
        server._running = False
        return ([], [], [])

    with patch("select.select", side_effect=toggle_not_ready):
        server._run_server()


def test_pipeline_engine_missing_branches():
    """Test missing branches in PipelineParallelismStrategy."""
    from ml_switcheroo_compiler.distributed.config_models import SchedulePhaseConfig
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    engine = PipelineParallelismStrategy(num_microbatches=2)

    # 560->563: barrier_id already in new_nodes
    g1 = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear", inputs=[])
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"])
    g1.nodes = {"n1": n1, "n2": n2}
    with patch.object(engine, "split_into_stages", return_value=[["n1", "n1"], ["n2"]]):
        with patch.object(engine, "insert_send_recv"):
            engine.unroll_pipeline(g1, num_stages=2)

    # 617->616 and 622->629: insert_send_recv
    # 617->616: inp_id not in node_to_stage ("external_inp")
    # 622->629: multiple consumers in target stage for same inp_id
    g2 = IRGraph()
    s0_node = IRNode(id="s0_out", op_type="Linear", inputs=[])
    s1_consumer_a = IRNode(id="s1_cA", op_type="Linear", inputs=["s0_out", "external_inp"])
    s1_consumer_b = IRNode(id="s1_cB", op_type="Linear", inputs=["s0_out"])
    g2.nodes = {"s0_out": s0_node, "s1_cA": s1_consumer_a, "s1_cB": s1_consumer_b}
    engine.insert_send_recv(g2, [["s0_out"], ["s1_cA", "s1_cB"]])

    # 669->668: in_id != inp.id in generate_microbatch_loop
    # 688->exit: graph.outputs is empty
    g3 = IRGraph()
    inp1 = IRNode(id="inp1", op_type="Input", inputs=[])
    inp2 = IRNode(id="inp2", op_type="Input", inputs=[])
    comp = IRNode(id="comp", op_type="Add", inputs=["inp1", "inp2"])
    g3.nodes = {"inp1": inp1, "inp2": inp2, "comp": comp}
    g3.outputs = []
    engine.generate_microbatch_loop(g3)

    # 739->724: custom phase type in generate_schedule
    engine.config.schedule.phases.append(SchedulePhaseConfig(type="custom_unknown", count_expression="1", operations=["custom_op"]))
    sched = engine.generate_schedule(IRGraph())
    assert isinstance(sched, list)

    # 773->775: lower with graph having hasattr(graph, 'attributes') == True
    g4 = IRGraph()
    g4.attributes = {"existing": 123}
    for i in range(12):
        g4.nodes[f"node_{i}"] = IRNode(id=f"node_{i}", op_type="Identity", inputs=[])
    res = engine.lower(g4)
    assert res is True
    assert g4.attributes["existing"] == 123
    assert "pipeline_schedule" in g4.attributes
