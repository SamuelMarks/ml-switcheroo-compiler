def test_strategy_resolvers():
    from unittest.mock import patch

    from ml_switcheroo_compiler.distributed.config_models import (
        load_cluster_topology,
        load_distributed_topologies,
    )
    from ml_switcheroo_compiler.distributed.strategy import (
        Coordinator,
        DataParallelStrategy,
        ModelParallelStrategy,
        SPMDShardingStrategy,
        _load_strategy_config,
        _load_webrtc_topology,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    c = Coordinator()
    assert not c.joined
    c.join()
    assert c.joined

    # Cluster topology and distributed topologies
    cluster_top = load_cluster_topology()
    assert "default" in cluster_top.cluster_meshes

    dist_top = load_distributed_topologies()
    assert "mesh_dp_tp" in dist_top.cluster_meshes
    assert "host_0_cost_matrix" in dist_top.communication_cost_matrices

    with patch("os.path.exists", return_value=False):
        assert _load_webrtc_topology() == {}
        assert _load_strategy_config() == {}

    dp = DataParallelStrategy(mesh_axis="dp")
    assert dp.mesh_axis == "dp"
    assert dp.get_communication_protocol() == "tcp"

    mp = ModelParallelStrategy(mesh_axis="tp")
    assert mp.mesh_axis == "tp"

    m = SPMDShardingStrategy(layout_map={"n_out": "spec"})
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
        ModelParallelStrategy,
        PerWorkerValue,
        PipelineParallelismStrategy,
        PreemptionCheckpointHandler,
        RemoteValue,
    )
    from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

    # ModelParallelStrategy
    mp = ModelParallelStrategy(mesh_axis="tp", row_parallel_dim=0, col_parallel_dim=1)
    g_mp = IRGraph()
    n_in = IRNode(id="x", op_type="Input")
    n_lin = IRNode(id="linear", op_type="Linear", inputs=["x"])
    n_act = IRNode(id="relu", op_type="Relu", inputs=["linear"])
    g_mp.nodes = {"x": n_in, "linear": n_lin, "relu": n_act}
    assert mp.partition_linear(g_mp) is True
    assert "linear_tp_all_reduce" in g_mp.nodes

    # PreemptionCheckpointHandler
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_cp_dir:
        p = PreemptionCheckpointHandler("res", tmp_cp_dir)
        assert p.cluster_resolver == "res"
        assert p.checkpoint_dir == tmp_cp_dir
        assert p.restore_latest_checkpoint() is None

        cp_path = p.save_preemption_checkpoint({"step": 42}, "preempt_cp_test")
        assert os.path.exists(cp_path)

        restored = p.restore_latest_checkpoint()
        assert restored is not None
        assert restored["checkpoint_id"] == "preempt_cp_test"
        assert restored["data"]["step"] == 42

        p.register_signal_handlers()
        with patch("signal.signal", side_effect=ValueError):
            p.register_signal_handlers()
        p._handle_preemption_signal(15, None)
        latest = p.restore_latest_checkpoint()
        assert latest is not None
        assert latest["data"]["signal"] == 15

    with tempfile.TemporaryDirectory() as empty_dir:
        p_empty = PreemptionCheckpointHandler("res", empty_dir)
        assert p_empty.restore_latest_checkpoint() is None

    p_nonexist = PreemptionCheckpointHandler("res", "/non/existent/dir/for/testing")
    assert p_nonexist.restore_latest_checkpoint() is None

    from ml_switcheroo_compiler.distributed.config_models import (
        load_distributed_topologies,
    )

    dt_cfg = load_distributed_topologies()
    assert "mesh_dp_tp" in dt_cfg.cluster_meshes
    assert "host_0" in dt_cfg.device_topologies
    assert "default_webrtc" in dt_cfg.signaling_topologies

    explicit_dt_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "distributed", "distributed_topologies.yaml")
    if os.path.exists(explicit_dt_path):
        dt_cfg2 = load_distributed_topologies(path=explicit_dt_path)
        assert "mesh_dp_tp" in dt_cfg2.cluster_meshes

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
