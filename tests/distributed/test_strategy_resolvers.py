def test_strategy_resolvers():
    import os
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.distributed.strategy import (
        CentralStorageStrategy,
        Coordinator,
        KubernetesClusterResolver,
        MeshShardingStrategy,
        SlurmClusterResolver,
        TFConfigClusterResolver,
        TPUStrategy,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    c = Coordinator()
    assert not c.joined
    c.join()
    assert c.joined

    with patch.dict(os.environ, {"TF_CONFIG": '{"cluster": {"worker": ["localhost:8080"]}}'}):
        tf = TFConfigClusterResolver()
        assert tf.cluster == {"worker": ["localhost:8080"]}

    with patch.dict(os.environ, {"TF_CONFIG": "invalid_json"}):
        with pytest.warns(UserWarning):
            tf = TFConfigClusterResolver()

    with patch.dict(os.environ, {}, clear=True):
        tf = TFConfigClusterResolver()
        assert tf.cluster == {}

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_NAME": "my_svc"}):
        with patch("socket.gethostbyname_ex", return_value=(None, None, ["127.0.0.1"])):
            k = KubernetesClusterResolver()
            assert k.cluster == {"worker": ["127.0.0.1:8080"]}

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_NAME": "my_svc"}):
        with patch("socket.gethostbyname_ex", side_effect=OSError("mock")):
            k = KubernetesClusterResolver()
            assert k.cluster == {"worker": ["localhost:8080"]}

    with patch.dict(os.environ, {"KUBERNETES_SERVICE_HOST": "1.2.3.4"}):
        k = KubernetesClusterResolver()
        assert len(k.cluster["worker"]) == 1

    with patch.dict(os.environ, {}, clear=True):
        k = KubernetesClusterResolver()
        assert k.cluster == {"worker": ["localhost:8080"]}

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node[01-03,05]"}):
        s = SlurmClusterResolver()
        assert s.cluster == {"worker": ["node01", "node02", "node03", "node05"]}

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": "node1,node2"}):
        s = SlurmClusterResolver()
        assert s.cluster == {"worker": ["node1", "node2"]}

    with patch.dict(os.environ, {"SLURM_JOB_NODELIST": ""}):
        s = SlurmClusterResolver()
        assert s.cluster == {}

    with patch("os.path.exists", return_value=False):
        from ml_switcheroo_compiler.distributed.strategy import _load_strategy_config, _load_webrtc_topology

        assert _load_webrtc_topology() == {}
        assert _load_strategy_config() == {}

    c = CentralStorageStrategy()
    assert c.config == {}

    t = TPUStrategy()
    assert t.config == {}

    m = MeshShardingStrategy(layout_map={"n_out": "spec"})
    g_mesh = IRGraph()
    n_in_mesh = IRNode(id="n_in_mesh", op_type="Input")
    n_in_mesh.sharding = "in_spec"
    n_out_mesh = IRNode(id="n_out_mesh", op_type="Add", inputs=["n_in_mesh"])
    g_mesh.nodes = {"n_in_mesh": n_in_mesh, "n_out_mesh": n_out_mesh}
    m.propagate_layouts(g_mesh)
    assert g_mesh.nodes["n_out_mesh"].sharding == "in_spec"

    n_out_mesh2 = IRNode(id="n_out", op_type="Add", inputs=["n_in_mesh"])
    g_mesh.nodes["n_out"] = n_out_mesh2
    m.propagate_layouts(g_mesh)
    assert g_mesh.nodes["n_out"].sharding == "spec"

    with patch("ml_switcheroo_compiler.transforms.passes.spmd.inject_spmd_communication_pass", return_value=True):
        assert m.lower_sharding(g_mesh) is True


def test_strategy_more_coverage():
    from unittest.mock import MagicMock, patch

    import pytest

    from ml_switcheroo_compiler.distributed.strategy import (
        CentralStorageStrategy,
        PerWorkerValue,
        PipelineParallelismStrategy,
        PreemptionCheckpointHandler,
        RemoteValue,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # CentralStorageStrategy
    c = CentralStorageStrategy()
    assert c.fetch() is None
    assert c.update() is None

    c.config = {"registry_hooks": {"fetch": "mock_fetch", "update": "mock_update"}}

    class MockBackendCS:
        def mock_fetch(self, *args, **kwargs):
            return "fetched"

        def mock_update(self, *args, **kwargs):
            return "updated"

    with patch("ml_switcheroo_compiler.backends.registry.get_active_backend", return_value=MockBackendCS()):
        assert c.fetch() == "fetched"
        assert c.update() == "updated"

    # PreemptionCheckpointHandler
    p = PreemptionCheckpointHandler("res", "dir")
    assert p.cluster_resolver == "res"
    assert p.checkpoint_dir == "dir"

    # PerWorkerValue
    pwv = PerWorkerValue([1, 2])
    assert pwv.values == [1, 2]

    # RemoteValue
    rv = RemoteValue()
    assert rv.value is None

    # PipelineParallelismStrategy
    with patch("ml_switcheroo_compiler.distributed.config_models.PipelineTopologiesConfig") as mock_conf:
        mock_conf.return_value.get.return_value = None
        with pytest.raises(ValueError):
            PipelineParallelismStrategy(topology_name="missing_topology")

    p = PipelineParallelismStrategy(num_microbatches=1)
    g = IRGraph()
    p.generate_microbatch_loop(g)  # returns early

    with pytest.raises(ValueError):
        p.split_into_stages(g, 0)

    p.num_microbatches = 2
    n = IRNode(id="n1", op_type="Add")
    g.nodes["n1"] = n
    # Make sure we hit the "continue" in unroll_pipeline
    with patch.object(p, "split_into_stages", return_value=[["missing_id"]]):
        p.unroll_pipeline(g, 1)
    from unittest.mock import patch

    import pytest

    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    strategy = PipelineParallelismStrategy(num_microbatches=4)
    g = IRGraph()
    # Add input and compute nodes for microbatch loop
    n_in = IRNode(id="n_in", op_type="Input")
    n_out = IRNode(id="n_out", op_type="Add", inputs=["n_in"])
    g.nodes = {"n_in": n_in, "n_out": n_out}
    g.inputs = ["n_in"]
    g.outputs = ["n_out"]

    # Needs to not have 'schedule' in config
    strategy.config = MagicMock()
    strategy.config.schedule = None

    strategy.generate_microbatch_loop(g)
    assert "microbatch_loop" in g.nodes
    assert "n_out_concat" in g.nodes

    schedule = strategy.generate_schedule(g)
    assert len(schedule) > 0
    assert schedule[0] == ("forward", 0)

    # Test tracking gradient bounds
    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input")
    n_grad = IRNode(id="n_grad", op_type="Grad", inputs=["n_in2"])
    n_opt = IRNode(id="n_opt", op_type="OptimizerStep", inputs=["n_grad"])
    g2.nodes = {"n_in2": n_in2, "n_grad": n_grad, "n_opt": n_opt}
    strategy.track_gradient_accumulation(g2)
    assert "n_grad_accum" in g2.nodes

    # Coverage for get_communication_protocol
    strategy.target_env = "browser"
    assert strategy.get_communication_protocol() == "webrtc"
    strategy.target_env = "host"
    assert strategy.get_communication_protocol() == "tcp"
