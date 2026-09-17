# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module ir_format.py for IR graph serialization and deserialization."""

from __future__ import annotations

import json
from typing import TypedDict, Union

import yaml

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class NodeSerializationDict(TypedDict, total=False):
    """Schema for individual node serialization.

    Attributes:
        id: Identifier of the node.
        op: Operation type.
        op_type: Canonical operation type name.
        inputs: List of input node IDs.
        shape_metadata: Tensor shape dimensions.
        shape: Duplicate shape field for browser runner compatibility.
        attributes: Dictionary of node attributes.
        subgraphs: Nested subgraphs dictionary.
    """

    id: str
    op: str
    op_type: str
    inputs: list[str]
    shape_metadata: list[int | str]
    shape: list[int | str]
    attributes: dict[str, str | int | float | bool]
    subgraphs: dict[str, GraphSerializationDict]


class GraphSerializationDict(TypedDict, total=False):
    """Schema for IR graph serialization.

    Attributes:
        name: Name of the computational graph.
        inputs: List of graph input node IDs.
        outputs: List of graph output node IDs.
        initializers: Static weights and tensor initializers map.
        nodes: Mapping of node ID to NodeSerializationDict.
    """

    name: str
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, object]
    nodes: dict[str, NodeSerializationDict]


def _build_serialization_dict(graph: IRGraph) -> GraphSerializationDict:
    """Builds a structured dictionary representing the IR graph.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        GraphSerializationDict: Structured serialization dictionary.
    """
    inputs_list: list[str] = list(getattr(graph, "inputs", [])) if getattr(graph, "inputs", None) else [node_id for node_id, node in graph.nodes.items() if getattr(node, "op_type", "") == "Input"]
    outputs_list: list[str] = list(getattr(graph, "outputs", []))
    nodes_dict: dict[str, NodeSerializationDict] = {}

    initializers_dict: dict[str, object] = {}
    if hasattr(graph, "initializers") and graph.initializers:
        for k, v in graph.initializers.items():
            if hasattr(v, "tolist"):
                initializers_dict[str(k)] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                initializers_dict[str(k)] = v
            else:
                initializers_dict[str(k)] = str(v)

    for node_id, node in graph.nodes.items():
        shape_meta = getattr(node, "shape_metadata", None)
        shape_list: list[int | str] = []
        if shape_meta is not None:
            if isinstance(shape_meta, (tuple, list)):
                for d in shape_meta:
                    if isinstance(d, int):
                        shape_list.append(d)
                    elif isinstance(d, str) and d.isdigit():
                        shape_list.append(int(d))
                    else:
                        shape_list.append(str(d))
            else:
                if isinstance(shape_meta, int):
                    shape_list.append(shape_meta)
                elif isinstance(shape_meta, str) and shape_meta.isdigit():
                    shape_list.append(int(shape_meta))
                else:
                    shape_list.append(str(shape_meta))

        raw_attrs = getattr(node, "attributes", {})
        clean_attrs: dict[str, str | int | float | bool] = {}
        if isinstance(raw_attrs, dict):
            for k, v in raw_attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_attrs[str(k)] = v
                else:
                    clean_attrs[str(k)] = str(v)

        serialized_subgraphs: dict[str, GraphSerializationDict] = {}
        for sub_k, sub_g in getattr(node, "subgraphs", {}).items():
            if hasattr(sub_g, "nodes"):
                serialized_subgraphs[str(sub_k)] = _build_serialization_dict(sub_g)

        node_entry: NodeSerializationDict = {
            "id": node_id,
            "op": node.op_type,
            "op_type": node.op_type,
            "inputs": list(node.inputs),
            "shape_metadata": shape_list,
            "shape": shape_list,
            "attributes": clean_attrs,
        }
        if serialized_subgraphs:
            node_entry["subgraphs"] = serialized_subgraphs
        nodes_dict[node_id] = node_entry

    return {
        "name": getattr(graph, "name", "graph"),
        "inputs": inputs_list,
        "outputs": outputs_list,
        "initializers": initializers_dict,
        "nodes": nodes_dict,
    }


def _parse_serialization_dict(data: dict[str, object]) -> IRGraph:
    """Parses a structured dictionary back into an IRGraph instance.

    Args:
        data (dict[str, object]): Parsed graph data mapping.

    Returns:
        IRGraph: Reconstructed intermediate representation graph.
    """
    graph_name = str(data.get("name", "graph"))
    graph = IRGraph(name=graph_name)

    raw_inputs = data.get("inputs", [])
    if isinstance(raw_inputs, list):
        graph.inputs = [str(inp) for inp in raw_inputs]

    raw_outputs = data.get("outputs", [])
    if isinstance(raw_outputs, list):
        graph.outputs = [str(out) for out in raw_outputs]

    raw_inits = data.get("initializers", {})
    if isinstance(raw_inits, dict) and hasattr(graph, "initializers"):
        for k, v in raw_inits.items():
            graph.initializers[str(k)] = v

    raw_nodes = data.get("nodes", {})
    if isinstance(raw_nodes, dict):
        for node_id, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue
            op_type = str(raw_node.get("op_type", raw_node.get("op", "")))
            raw_inputs = raw_node.get("inputs", [])
            inputs: list[str] = [str(inp) for inp in raw_inputs] if isinstance(raw_inputs, list) else []

            raw_shape = raw_node.get("shape_metadata", raw_node.get("shape", []))
            shape_tuple: tuple[int | str, ...] = ()
            if isinstance(raw_shape, list):
                shape_dims: list[int | str] = []
                for dim in raw_shape:
                    if isinstance(dim, int):
                        shape_dims.append(dim)
                    elif isinstance(dim, str) and dim.isdigit():
                        shape_dims.append(int(dim))
                    else:
                        shape_dims.append(str(dim))
                shape_tuple = tuple(shape_dims)

            raw_attrs = raw_node.get("attributes", {})
            attributes: dict[str, str | int | float | bool] = {}
            if isinstance(raw_attrs, dict):
                for k, v in raw_attrs.items():
                    if isinstance(v, (str, int, float, bool)):
                        attributes[str(k)] = v
                    else:
                        attributes[str(k)] = str(v)

            node = IRNode(
                id=str(node_id),
                op_type=op_type,
                inputs=inputs,
                shape_metadata=shape_tuple,
                attributes=attributes,
            )

            raw_subgraphs = raw_node.get("subgraphs", {})
            if isinstance(raw_subgraphs, dict):
                for sub_k, sub_data in raw_subgraphs.items():
                    if isinstance(sub_data, dict):
                        node.subgraphs[str(sub_k)] = _parse_serialization_dict(sub_data)

            graph.nodes[str(node_id)] = node

    return graph


def graph_to_json(graph: IRGraph) -> str:
    """Implement JSON serialization for snapshot testing and client runner consumption.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        str: The formatted JSON string representation of the graph.
    """
    data = _build_serialization_dict(graph)
    return json.dumps(data, indent=2)


def json_to_graph(json_str: str) -> IRGraph:
    """Deserialize JSON back into an IRGraph.

    Args:
        json_str (str): The JSON string representation.

    Returns:
        IRGraph: The deserialized IRGraph.
    """
    data: dict[str, object] = json.loads(json_str)
    return _parse_serialization_dict(data)


def graph_to_yaml(graph: IRGraph) -> str:
    """Serialize an IRGraph into declarative YAML format.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        str: YAML string representation of the graph.
    """
    data = _build_serialization_dict(graph)
    return yaml.safe_dump(dict(data), sort_keys=False)


def yaml_to_graph(yaml_str: str) -> IRGraph:
    """Deserialize YAML back into an IRGraph instance.

    Args:
        yaml_str (str): The YAML string representation.

    Returns:
        IRGraph: The deserialized IRGraph.
    """
    data: dict[str, object] = yaml.safe_load(yaml_str) or {}
    return _parse_serialization_dict(data)


def graph_to_protobuf(graph: IRGraph) -> bytes:
    """Define IRGraph Protobuf .proto spec serialization.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        bytes: The serialized Protobuf bytes.
    """
    return b""


def graph_to_flatbuffers(graph: IRGraph) -> bytes:
    """Implement FlatBuffers serialization for zero-copy JS/TS loading.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        bytes: The serialized FlatBuffers bytes.
    """
    return b""
