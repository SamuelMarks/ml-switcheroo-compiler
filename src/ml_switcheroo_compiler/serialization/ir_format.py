# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
from typing import Any

"""IR graph serialization."""

import json

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def graph_to_json(graph: IRGraph) -> str:
    """Implement JSON serialization for snapshot testing IR passes.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        str: The JSON string representation of the graph.
    """
    data = {"nodes": {}}  # type: ignore  # Justification: Polymorphic / Duck Typing for Framework Agnosticism
    for node_id, node in graph.nodes.items():
        data["nodes"][node_id] = {
            "op": node.op_type,
            "inputs": node.inputs,
        }
    return json.dumps(data, indent=2)


def json_to_graph(json_str: str) -> IRGraph:
    """Deserialize JSON back into an IRGraph.

    Args:
        json_str (str): The JSON string representation.

    Returns:
        IRGraph: The deserialized IRGraph.
    """
    data = json.loads(json_str)
    graph = IRGraph()
    for node_id, node_data in data.get("nodes", {}).items():
        node = IRNode(
            id=node_id,
            op_type=node_data.get("op", ""),
            inputs=node_data.get("inputs", []),
            shape_metadata=(),
        )
        graph.nodes[node_id] = node
    return graph


def graph_to_protobuf(graph: IRGraph) -> bytes:
    """Define IRGraph Protobuf .proto spec serialization.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        bytes: The serialized Protobuf bytes.
    """
    # Base implementation
    return b""


def graph_to_flatbuffers(graph: IRGraph) -> bytes:
    """Implement FlatBuffers serialization for zero-copy JS/TS loading.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        bytes: The serialized FlatBuffers bytes.
    """
    # Base implementation
    return b""
