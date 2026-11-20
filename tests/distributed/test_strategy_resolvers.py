"""Tests for test_strategy_resolvers."""

from __future__ import annotations

import types
import typing

import pytest
from ml_switcheroo_ir import NodeDict

from tests.testing_utils import EagerBoundCallable, require_callable

if typing.TYPE_CHECKING:
    from unittest.mock import MagicMock

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.distributed.strategy import (
        PerWorkerValue,
        PipelineParallelStrategy,
        PreemptionCheckpointHandler,
        RemoteValue,
    )


def test_strategy_resolvers() -> None:
    """Test function test_strategy_resolvers."""
    from unittest.mock import patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.distributed.config_models import load_cluster_topology, load_distributed_topologies
    from ml_switcheroo_compiler.distributed.strategy import (
        Coordinator,
        DataParallelStrategy,
        ModelParallelStrategy,
        SPMDShardingStrategy,
        _load_strategy_config,
        _load_webrtc_topology,
    )

    c = Coordinator()
    joined_init: bool = c.joined
    assert not joined_init
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
    g_mesh.nodes = NodeDict({"n_in_mesh": n_in_mesh, "n_out_mesh": n_out_mesh})
    m.propagate_layouts(g_mesh)
    assert g_mesh.nodes["n_out_mesh"].sharding == "in_spec"

    n_out_mesh2 = IRNode(id="n_out", op_type="Add", inputs=["n_in_mesh"])
    g_mesh.nodes["n_out"] = n_out_mesh2
    m.propagate_layouts(g_mesh)
    assert g_mesh.nodes["n_out"].sharding == "spec"

    with patch("ml_switcheroo_compiler.transforms.passes.spmd.inject_spmd_communication_pass", return_value=True):
        assert m.lower_sharding(g_mesh) is True


class _TestStrategyMoreCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_strategy_more_coverage` split helpers."""

    IRGraph: type[IRGraph]
    IRNode: type[IRNode]
    MagicMock: type[MagicMock]
    PerWorkerValue: type[PerWorkerValue]
    PipelineParallelismStrategy: type[PipelineParallelStrategy]
    PreemptionCheckpointHandler: type[PreemptionCheckpointHandler]
    RemoteValue: type[RemoteValue]
    g: IRGraph
    os: types.ModuleType
    p: PipelineParallelStrategy
    p_nonexist: PreemptionCheckpointHandler
    patch: EagerBoundCallable
    strategy: PipelineParallelStrategy
    tempfile: types.ModuleType


def _test_strategy_more_coverage_part_1(_state: _TestStrategyMoreCoverageState) -> None:
    """Execute part 1 of test_strategy_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import MagicMock, patch

    from ml_switcheroo_ir import IRGraph, IRNode

    from ml_switcheroo_compiler.distributed.strategy import (
        ModelParallelStrategy,
        PerWorkerValue,
        PipelineParallelismStrategy,
        PreemptionCheckpointHandler,
        RemoteValue,
    )

    # ModelParallelStrategy
    mp = ModelParallelStrategy(mesh_axis="tp", row_parallel_dim=0, col_parallel_dim=1)
    g_mp = IRGraph()
    n_in = IRNode(id="x", op_type="Input")
    n_lin = IRNode(id="linear", op_type="Linear", inputs=["x"])
    n_act = IRNode(id="relu", op_type="Relu", inputs=["linear"])
    g_mp.nodes = NodeDict({"x": n_in, "linear": n_lin, "relu": n_act})
    assert mp.partition_linear(g_mp) is True
    assert "linear_tp_all_reduce" in g_mp.nodes

    # PreemptionCheckpointHandler
    import os
    import tempfile

    _state.update({"IRGraph": IRGraph, "IRNode": IRNode, "MagicMock": MagicMock})
    _state.update(
        {
            "PerWorkerValue": PerWorkerValue,
            "PipelineParallelismStrategy": PipelineParallelismStrategy,
            "PreemptionCheckpointHandler": PreemptionCheckpointHandler,
            "RemoteValue": RemoteValue,
        }
    )
    _state.update({"os": os, "patch": require_callable(patch)})
    _state.update({"tempfile": tempfile})


def _test_strategy_more_coverage_part_2(_state: _TestStrategyMoreCoverageState) -> None:
    """Execute part 2 of test_strategy_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    PreemptionCheckpointHandler, os, patch, tempfile = (
        _state["PreemptionCheckpointHandler"],
        _state["os"],
        _state["patch"],
        _state["tempfile"],
    )
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
        restored_data = restored["data"]
        assert isinstance(restored_data, typing.Mapping)
        assert restored_data["step"] == 42

        p.register_signal_handlers()
        with patch("signal.signal", side_effect=ValueError):
            p.register_signal_handlers()
        p._handle_preemption_signal(15, None)
        latest = p.restore_latest_checkpoint()
        assert latest is not None
        latest_data = latest["data"]
        assert isinstance(latest_data, typing.Mapping)
        assert latest_data["signal"] == 15

    with tempfile.TemporaryDirectory() as empty_dir:
        p_empty = PreemptionCheckpointHandler("res", empty_dir)
        assert p_empty.restore_latest_checkpoint() is None

    p_nonexist = PreemptionCheckpointHandler("res", "/non/existent/dir/for/testing")
    _state.update({"p_nonexist": p_nonexist})
    _state.update(
        {"PreemptionCheckpointHandler": PreemptionCheckpointHandler, "os": os, "patch": require_callable(patch)}
    )
    _state.update({"tempfile": tempfile})


def _test_strategy_more_coverage_part_3(_state: _TestStrategyMoreCoverageState) -> None:
    """Execute part 3 of test_strategy_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, PerWorkerValue, PipelineParallelismStrategy, RemoteValue = (
        _state["IRGraph"],
        _state["PerWorkerValue"],
        _state["PipelineParallelismStrategy"],
        _state["RemoteValue"],
    )
    os, p_nonexist, patch = _state["os"], _state["p_nonexist"], _state["patch"]
    assert p_nonexist.restore_latest_checkpoint() is None

    from ml_switcheroo_compiler.distributed.config_models import load_distributed_topologies

    dt_cfg = load_distributed_topologies()
    assert "mesh_dp_tp" in dt_cfg.cluster_meshes
    assert "host_0" in dt_cfg.device_topologies
    assert "default_webrtc" in dt_cfg.signaling_topologies

    explicit_dt_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "src",
        "ml_switcheroo_compiler",
        "distributed",
        "distributed_topologies.yaml",
    )
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
        pytest.MonkeyPatch().setattr(mock_conf.return_value.get, "return_value", None, raising=False)
        with pytest.raises(ValueError):
            PipelineParallelismStrategy(topology_name="missing_topology")

    p = PipelineParallelismStrategy(num_microbatches=1)
    g = IRGraph()
    p.generate_microbatch_loop(g)  # returns early
    _state.update({"g": g})
    _state.update({"p": p})
    _state.update(
        {
            "IRGraph": IRGraph,
            "PerWorkerValue": PerWorkerValue,
            "PipelineParallelismStrategy": PipelineParallelismStrategy,
        }
    )
    _state.update({"RemoteValue": RemoteValue, "os": os, "p_nonexist": p_nonexist})
    _state.update({"patch": require_callable(patch)})


def _test_strategy_more_coverage_part_4(_state: _TestStrategyMoreCoverageState) -> None:
    """Execute part 4 of test_strategy_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    from unittest.mock import patch

    IRNode, MagicMock, g, p = _state["IRNode"], _state["MagicMock"], _state["g"], _state["p"]
    with pytest.raises(ValueError):
        p.split_into_stages(g, 0)

    pytest.MonkeyPatch().setattr(p, "num_microbatches", 2, raising=False)
    n = IRNode(id="n1", op_type="Add")
    g.nodes["n1"] = n
    # Make sure we hit the "continue" in unroll_pipeline
    with patch.object(p, "split_into_stages", return_value=[["missing_id"]]):
        p.unroll_pipeline(g, 1)
    from ml_switcheroo_ir import IRGraph

    from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy

    strategy = PipelineParallelismStrategy(num_microbatches=4)
    g = IRGraph()
    # Add input and compute nodes for microbatch loop
    n_in = IRNode(id="n_in", op_type="Input")
    n_out = IRNode(id="n_out", op_type="Add", inputs=["n_in"])
    pytest.MonkeyPatch().setattr(g, "nodes", NodeDict({"n_in": n_in, "n_out": n_out}), raising=False)
    pytest.MonkeyPatch().setattr(g, "inputs", ["n_in"], raising=False)
    pytest.MonkeyPatch().setattr(g, "outputs", ["n_out"], raising=False)

    # Needs to not have 'schedule' in config
    pytest.MonkeyPatch().setattr(strategy, "config", MagicMock(), raising=False)
    strategy.config.schedule = None

    strategy.generate_microbatch_loop(g)
    assert "microbatch_loop" in g.nodes
    _state.update(
        {"IRGraph": IRGraph, "IRNode": IRNode, "PipelineParallelismStrategy": PipelineParallelismStrategy, "g": g}
    )
    _state.update({"patch": require_callable(patch)})
    _state.update({"strategy": strategy})
    _state.update({"MagicMock": MagicMock, "p": p})


def _test_strategy_more_coverage_part_5(_state: _TestStrategyMoreCoverageState) -> None:
    """Execute part 5 of test_strategy_more_coverage.

    Args:
        _state: Shared test state dictionary.
    """
    IRGraph, IRNode, g, strategy = _state["IRGraph"], _state["IRNode"], _state["g"], _state["strategy"]
    assert "n_out_concat" in g.nodes

    schedule = strategy.generate_schedule(g)
    assert len(schedule) > 0
    assert schedule[0] == ("forward", 0)

    # Test tracking gradient bounds
    g2 = IRGraph()
    n_in2 = IRNode(id="n_in2", op_type="Input")
    n_grad = IRNode(id="n_grad", op_type="Grad", inputs=["n_in2"])
    n_opt = IRNode(id="n_opt", op_type="OptimizerStep", inputs=["n_grad"])
    pytest.MonkeyPatch().setattr(
        g2, "nodes", NodeDict({"n_in2": n_in2, "n_grad": n_grad, "n_opt": n_opt}), raising=False
    )
    strategy.track_gradient_accumulation(g2)
    assert "n_grad_accum" in g2.nodes

    # Coverage for get_communication_protocol
    pytest.MonkeyPatch().setattr(strategy, "target_env", "browser", raising=False)
    assert strategy.get_communication_protocol() == "webrtc"
    pytest.MonkeyPatch().setattr(strategy, "target_env", "host", raising=False)
    assert strategy.get_communication_protocol() == "tcp"


def test_strategy_more_coverage() -> None:
    """Test function test_strategy_more_coverage."""
    _state: _TestStrategyMoreCoverageState = {}
    _test_strategy_more_coverage_part_1(_state)
    _test_strategy_more_coverage_part_2(_state)
    _test_strategy_more_coverage_part_3(_state)
    _test_strategy_more_coverage_part_4(_state)
    _test_strategy_more_coverage_part_5(_state)
    assert len(_state) > 0
