"""IR graph serialization."""

import json

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def graph_to_json(graph: IRGraph) -> str:
    """Implement JSON serialization for snapshot testing IR passes.

    Args:
        graph (IRGraph): Argument graph

    Returns:
        str: The evaluated output resulting from this operation.
    """
    data = {"nodes": {}}
    for node_id, node in graph.nodes.items():
        data["nodes"][node_id] = {
            "op": node.op_type,
            "inputs": node.inputs,
        }
    return json.dumps(data, indent=2)


def json_to_graph(json_str: str) -> IRGraph:
    """Deserialize JSON back into an IRGraph.

    Args:
        json_str (str): Argument json_str

    Returns:
        IRGraph: The evaluated output resulting from this operation.
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
        graph (IRGraph): Argument graph

    Returns:
        bytes: The evaluated output resulting from this operation.
    """
    # Base implementation
    return b""


def graph_to_flatbuffers(graph: IRGraph) -> bytes:
    """Implement FlatBuffers serialization for zero-copy JS/TS loading.

    Args:
        graph (IRGraph): Argument graph

    Returns:
        bytes: The evaluated output resulting from this operation.
    """
    # Base implementation
    return b""
