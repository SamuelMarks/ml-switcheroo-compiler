"""Tests for distributed pipeline parallelism primitives and topology configurations."""

from __future__ import annotations

import os

import pytest
import yaml

from ml_switcheroo_compiler.distributed.config_models import PipelineTopologiesConfig
from ml_switcheroo_compiler.distributed.strategy import PipelineParallelismStrategy
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_pipeline_parallelism_strategy_init_topologies() -> None:
    """Test initialization of PipelineParallelismStrategy with built-in topologies."""
    strat_default = PipelineParallelismStrategy(topology_name="default")
    assert strat_default.num_microbatches == 4
    assert strat_default.devices_per_stage == 1
    assert strat_default.strategy == "1f1b"
    assert strat_default.protocol == "tcp"

    strat_gpipe = PipelineParallelismStrategy(topology_name="gpipe", num_microbatches=16)
    assert strat_gpipe.num_microbatches == 16
    assert strat_gpipe.strategy == "sequential"
    assert strat_gpipe.protocol == "rpc"

    strat_pipedream = PipelineParallelismStrategy(topology_name="pipedream")
    assert strat_pipedream.strategy == "interleaved"
    assert strat_pipedream.protocol == "async_rpc"

    strat_webrtc = PipelineParallelismStrategy(topology_name="webrtc_pipeline")
    assert strat_webrtc.protocol == "webrtc"


def test_pipeline_parallelism_strategy_init_unknown_topology_fallback() -> None:
    """Test initialization with unknown topology falling back to default or raising."""
    strat = PipelineParallelismStrategy(topology_name="nonexistent_topology")
    assert strat.config is not None
    assert strat.strategy == "1f1b"


def test_split_into_stages_valid() -> None:
    """Test splitting an IRGraph into valid pipeline stages."""
    strat = PipelineParallelismStrategy()
    graph = IRGraph()
    for i in range(6):
        nid: str = f"node_{i}"
        graph.nodes[nid] = IRNode(id=nid, op_type="Relu", inputs=[f"node_{i - 1}"] if i > 0 else ["in"])

    stages = strat.split_into_stages(graph, num_stages=3)
    assert len(stages) == 3
    assert stages[0] == ["node_0", "node_1"]
    assert stages[1] == ["node_2", "node_3"]
    assert stages[2] == ["node_4", "node_5"]


def test_split_into_stages_invalid_count() -> None:
    """Test that split_into_stages raises ValueError when num_stages is non-positive."""
    strat = PipelineParallelismStrategy()
    graph = IRGraph()
    graph.nodes["n1"] = IRNode(id="n1", op_type="Identity")

    with pytest.raises(ValueError, match="Number of stages must be positive"):
        strat.split_into_stages(graph, num_stages=0)

    with pytest.raises(ValueError, match="Number of stages must be positive"):
        strat.split_into_stages(graph, num_stages=-2)


def test_insert_send_recv_nodes() -> None:
    """Test insertion of Send and Recv nodes across stage boundaries."""
    strat = PipelineParallelismStrategy()
    graph = IRGraph()

    # Stage 0 node
    graph.nodes["node_a"] = IRNode(id="node_a", op_type="Add", inputs=["in_0", "in_1"])
    # Stage 1 node depending on Stage 0 node
    graph.nodes["node_b"] = IRNode(id="node_b", op_type="Mul", inputs=["node_a", "in_2"])

    stages: list[list[str]] = [["node_a"], ["node_b"]]
    strat.insert_send_recv(graph, stages)

    assert "node_a_send_0_to_1" in graph.nodes
    assert "node_a_recv_0_to_1" in graph.nodes

    send_node = graph.nodes["node_a_send_0_to_1"]
    assert send_node.op_type == "Send"
    assert send_node.inputs == ["node_a"]
    assert send_node.attributes.get("target_stage") == 1

    recv_node = graph.nodes["node_a_recv_0_to_1"]
    assert recv_node.op_type == "Recv"
    assert recv_node.attributes.get("source_stage") == 0
    assert recv_node.attributes.get("target_stage") == 1


def test_pipeline_topologies_yaml_schema() -> None:
    """Validate pipeline_topologies.yaml against PipelineTopologiesConfig schema."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "..", "src", "ml_switcheroo_compiler", "distributed", "pipeline_topologies.yaml")
    with open(yaml_path, encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    config = PipelineTopologiesConfig(root=raw_data)
    assert "default" in config.root
    assert "1f1b" in config.root
    assert "gpipe" in config.root
