"""Test distributed strategy coverage."""

from unittest.mock import MagicMock

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.distributed.strategy import KubernetesClusterResolver, MeshShardingStrategy, Server, SlurmClusterResolver, TFConfigClusterResolver


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
