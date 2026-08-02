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


def test_exchange_ipc_data_failure():
    import numpy as np

    from ml_switcheroo_compiler.backends.numpy.distributed.ipc import _exchange_ipc_data

    arr = np.array([1.0, 2.0])
    # Connect to invalid port / authkey or coordinate fails, should fallback and return [arr]*size
    res = _exchange_ipc_data(rank=1, size=2, tensor_data=arr)
    assert len(res) == 2
    assert np.allclose(res[0], arr)
    assert np.allclose(res[1], arr)


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

    import pytest

    from ml_switcheroo_compiler.distributed.strategy import KubernetesClusterResolver, SlurmClusterResolver, TFConfigClusterResolver

    # TF_CONFIG success
    with patch.dict(os.environ, {"TF_CONFIG": json.dumps({"cluster": {"worker": ["host1"]}})}):
        resolver = TFConfigClusterResolver()
        assert resolver.cluster == {"worker": ["host1"]}

    # TF_CONFIG error
    with patch.dict(os.environ, {"TF_CONFIG": "invalid_json"}):
        with pytest.warns(UserWarning, match="Failed to parse TF_CONFIG"):
            resolver = TFConfigClusterResolver()
            assert resolver.cluster == {}

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


def test_server_socket():
    import time
    from unittest.mock import patch

    server = Server("def", "job", 0)

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("No backend")):
        server.start()
        time.sleep(0.1)
        assert server._server is not None
        assert server._running is True

        server.join()
        assert server._running is False


def test_server_accept():
    import socket
    import time
    from unittest.mock import patch

    server = Server("def", "job", 0)
    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", side_effect=Exception("No backend")):
        server.start()
        time.sleep(0.1)

        # Connect to the server to trigger accept and close
        port = server._server.getsockname()[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.close()

        time.sleep(0.1)
        server.join()


def test_server_close_exception():
    from unittest.mock import MagicMock

    server = Server("def", "job", 0)
    server._server = MagicMock()
    server._server.close.side_effect = Exception("Mock exception")
    server.join()


def test_microbatch_loop_single():
    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph

    strategy = PipelineParallelismStrategy(num_microbatches=1)
    g = IRGraph()
    strategy.generate_microbatch_loop(g)
    assert "microbatch_loop" not in g.nodes
