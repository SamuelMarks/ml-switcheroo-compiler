"""Docstring module."""

import json
from ml_switcheroo.ir.core import IRGraph, IRNode
from ml_switcheroo.serialization import (
    graph_to_json,
    json_to_graph,
    graph_to_protobuf,
    graph_to_flatbuffers,
)


def test_graph_to_json() -> None:
    """Docstring."""
    graph = IRGraph()
    node = IRNode(
        id="node1",
        op_type="TestOp",
        inputs=["input1"],
        shape_metadata=(),
    )
    graph.nodes["node1"] = node

    json_str = graph_to_json(graph)
    data = json.loads(json_str)

    assert "nodes" in data
    assert "node1" in data["nodes"]
    assert data["nodes"]["node1"]["op"] == "TestOp"
    assert data["nodes"]["node1"]["inputs"] == ["input1"]


def test_json_to_graph() -> None:
    """Docstring."""
    json_str = json.dumps(
        {"nodes": {"node2": {"op": "AnotherOp", "inputs": ["in1", "in2"]}}}
    )

    graph = json_to_graph(json_str)
    assert len(graph.nodes) == 1
    assert "node2" in graph.nodes
    node = graph.nodes["node2"]
    assert node.id == "node2"
    assert node.op_type == "AnotherOp"
    assert node.inputs == ["in1", "in2"]


def test_graph_to_protobuf() -> None:
    """Docstring."""
    graph = IRGraph()
    res = graph_to_protobuf(graph)
    assert res == b""


def test_graph_to_flatbuffers() -> None:
    """Docstring."""
    graph = IRGraph()
    res = graph_to_flatbuffers(graph)
    assert res == b""
