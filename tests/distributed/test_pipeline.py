"""Tests for PipelineParallelismStrategy."""

from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_pipeline_split_into_stages():
    """Test splitting graph into stages."""
    graph = IRGraph()
    for i in range(10):
        graph.nodes[f"node_{i}"] = IRNode(id=f"node_{i}", op_type="Linear")

    strategy = PipelineParallelismStrategy(num_microbatches=4, devices_per_stage=1)
    stages = strategy.split_into_stages(graph, num_stages=3)

    assert len(stages) == 3
    assert len(stages[0]) == 3
    assert len(stages[1]) == 3
    assert len(stages[2]) == 4


def test_pipeline_insert_send_recv():
    """Test inserting Send and Recv nodes."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Input")
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"])
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2

    strategy = PipelineParallelismStrategy(num_microbatches=4, devices_per_stage=1)
    stages = [["n1"], ["n2"]]
    strategy.insert_send_recv(graph, stages)

    # Expected: Send node inserted after n1, Recv node inserted before n2
    assert "n1_send_0_to_1" in graph.nodes
    assert "n1_recv_0_to_1" in graph.nodes
    assert graph.nodes["n2"].inputs == ["n1_recv_0_to_1"]


def test_pipeline_unroll_1f1b():
    """Test 1F1B unrolling logic."""
    graph = IRGraph()
    n1 = IRNode(id="n1", op_type="Linear")
    n2 = IRNode(id="n2", op_type="Linear", inputs=["n1"])
    graph.nodes["n1"] = n1
    graph.nodes["n2"] = n2
    graph.outputs = ["n2"]

    strategy = PipelineParallelismStrategy(num_microbatches=2, devices_per_stage=1)
    strategy.config.microbatch_splitting.strategy = "1f1b"
    strategy.unroll_pipeline(graph, num_stages=2)

    # Should have nodes for mb0 and mb1
    assert "n1_mb0" in graph.nodes
    assert "n1_mb1" in graph.nodes
    assert "n2_mb0" in graph.nodes
    assert "n2_mb1" in graph.nodes

    # Check 1F1B barrier was inserted
    assert "barrier_1_mb1" in graph.nodes
    assert "n2_mb1" in graph.nodes
    assert "barrier_1_mb1" in graph.nodes["n2_mb1"].inputs


def test_pipeline_track_gradient_accumulation():
    """Test gradient accumulation."""
    graph = IRGraph()
    g1 = IRNode(id="g1", op_type="Grad")
    graph.nodes["g1"] = g1

    strategy = PipelineParallelismStrategy()
    strategy.track_gradient_accumulation(graph)

    assert "g1_accum" in graph.nodes
    assert graph.nodes["g1_accum"].op_type == "Add"


def test_pipeline_lower():
    """Test overall lower method."""
    graph = IRGraph()
    for i in range(10):
        graph.nodes[f"node_{i}"] = IRNode(id=f"node_{i}", op_type="Linear")
    graph.outputs = ["node_9"]

    strategy = PipelineParallelismStrategy(num_microbatches=2, devices_per_stage=1)
    result = strategy.lower(graph)
    assert result is True
    assert "pipeline_schedule" in graph.attributes
    assert graph.attributes["num_pipeline_stages"] == 2
