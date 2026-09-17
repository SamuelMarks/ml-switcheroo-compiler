"""Tests for distributed pipeline parallelism runtime execution and scheduling."""

from __future__ import annotations

from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_unroll_pipeline_1f1b_schedule() -> None:
    """Test unrolling pipeline with 1F1B schedule and barrier dependencies."""
    strat = PipelineParallelismStrategy(topology_name="1f1b", num_microbatches=3)
    graph = IRGraph()

    # Stage 0: Input and Layer 1
    graph.nodes["layer_0"] = IRNode(id="layer_0", op_type="Linear", inputs=["in_x"])
    graph.nodes["layer_1"] = IRNode(id="layer_1", op_type="Relu", inputs=["layer_0"])

    # Stage 1: Layer 2 and Output
    graph.nodes["layer_2"] = IRNode(id="layer_2", op_type="Linear", inputs=["layer_1"])
    graph.nodes["loss"] = IRNode(id="loss", op_type="MSELoss", inputs=["layer_2", "targets"])
    graph.outputs = ["loss"]

    strat.unroll_pipeline(graph, num_stages=2)

    # Verify microbatch unrolling
    assert "loss_mb2" in graph.outputs

    # Verify barrier nodes exist for 1F1B microbatches > 0
    barrier_nodes = [nid for nid in graph.nodes if nid.startswith("barrier_")]
    assert len(barrier_nodes) > 0

    # Verify node instances per microbatch
    for mb in range(3):
        assert f"layer_0_mb{mb}" in graph.nodes
        assert f"loss_mb{mb}" in graph.nodes


def test_unroll_pipeline_sequential_schedule() -> None:
    """Test unrolling pipeline with sequential (gpipe) schedule without 1F1B barriers."""
    strat = PipelineParallelismStrategy(topology_name="gpipe", num_microbatches=2)
    graph = IRGraph()

    graph.nodes["op_a"] = IRNode(id="op_a", op_type="Add", inputs=["in_1", "in_2"])
    graph.nodes["op_b"] = IRNode(id="op_b", op_type="Mul", inputs=["op_a", "in_3"])
    graph.outputs = ["op_b"]

    strat.unroll_pipeline(graph, num_stages=2)

    # GPipe / sequential does not insert 1f1b barriers
    barrier_nodes = [nid for nid in graph.nodes if nid.startswith("barrier_")]
    assert len(barrier_nodes) == 0

    # Both microbatches exist
    assert "op_a_mb0" in graph.nodes
    assert "op_a_mb1" in graph.nodes
    assert "op_b_mb1" in graph.outputs


def test_unroll_pipeline_single_stage() -> None:
    """Test unrolling pipeline when num_stages is 1."""
    strat = PipelineParallelismStrategy(num_microbatches=2)
    graph = IRGraph()
    graph.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["x"])
    graph.outputs = ["conv"]

    strat.unroll_pipeline(graph, num_stages=1)
    assert "conv_mb0" in graph.nodes
    assert "conv_mb1" in graph.nodes
    assert graph.outputs == ["conv_mb1"]
