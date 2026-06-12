"""Unit tests for IRGraph serialization and deserialization.

This module contains test cases to verify the conversion of IRGraph objects to and from
various serialization formats, including JSON, Protobuf, and FlatBuffers.
"""

import json

from ml_switcheroo.ir.core import IRGraph, IRNode
from ml_switcheroo.serialization import (
    graph_to_flatbuffers,
    graph_to_json,
    graph_to_protobuf,
    json_to_graph,
)


def test_graph_to_json() -> None:
    """Verifies that an IRGraph can be successfully serialized into a JSON string.

    This test constructs an IRGraph with a single IRNode, serializes it using
    graph_to_json, and asserts that the resulting JSON structure contains the
    expected node properties such as operation type and inputs

    Returns:
    None.
    """
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
    """Verifies that a JSON string can be successfully deserialized into an IRGraph.

    This test defines a JSON string representing a graph, deserializes it using
    json_to_graph, and asserts that the reconstructed IRGraph contains the
    correct node attributes, including ID, operation type, and inputs

    Returns:
    None.
    """
    json_str = json.dumps(
        {"nodes": {"node2": {"op": "AnotherOp", "inputs": ["in1", "in2"]}}},
    )

    graph = json_to_graph(json_str)
    assert len(graph.nodes) == 1
    assert "node2" in graph.nodes
    node = graph.nodes["node2"]
    assert node.id == "node2"
    assert node.op_type == "AnotherOp"
    assert node.inputs == ["in1", "in2"]


def test_graph_to_protobuf() -> None:
    """Verifies that an IRGraph can be successfully serialized into Protobuf format.

    This test serializes an empty IRGraph and asserts that the resulting
    Protobuf byte representation is correct

    Returns:
    None.
    """
    graph = IRGraph()
    res = graph_to_protobuf(graph)
    assert res == b""


def test_graph_to_flatbuffers() -> None:
    """Verifies that an IRGraph can be successfully serialized into FlatBuffers format.

    This test serializes an empty IRGraph and asserts that the resulting
    FlatBuffers byte representation is correct

    Returns:
    None.
    """
    graph = IRGraph()
    res = graph_to_flatbuffers(graph)
    assert res == b""
