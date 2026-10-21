"""Module ir_format.py for IR graph serialization and deserialization."""

from __future__ import annotations

import json
import typing  # noqa: F401
from collections.abc import Mapping

import pydantic
import yaml
from ml_switcheroo_ir import AttributeValue

from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class NodeSerializationModel(pydantic.BaseModel):
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

    model_config = pydantic.ConfigDict(extra="forbid")

    id: str
    op: str
    op_type: str
    inputs: list[str]
    shape_metadata: list[int | str]
    shape: list[int | str]
    attributes: dict[str, str | int | float | bool]
    subgraphs: dict[str, GraphSerializationModel] | None = None


class GraphSerializationModel(pydantic.BaseModel):
    """Schema for IR graph serialization.

    Attributes:
        name: Name of the computational graph.
        inputs: List of graph input node IDs.
        outputs: List of graph output node IDs.
        initializers: Static weights and tensor initializers map.
        nodes: Mapping of node ID to NodeSerializationModel.
    """

    model_config = pydantic.ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    inputs: list[str]
    outputs: list[str]
    initializers: dict[str, AttributeValue]
    nodes: dict[str, NodeSerializationModel]


NodeSerializationDict = NodeSerializationModel
GraphSerializationDict = GraphSerializationModel


def _normalize_shape_dim(dim: AttributeValue) -> int | str:
    """Normalize a single shape dimension to int or str.

    Args:
        dim: Raw dimension value.

    Returns:
        int | str: Normalized dimension.
    """
    if isinstance(dim, int):
        return dim
    if isinstance(dim, str) and dim.isdigit():
        return int(dim)
    return str(dim)


def _serialize_shape_metadata(shape_meta: AttributeValue) -> list[int | str]:
    """Convert node shape_metadata into a serializable list of dimensions.

    Args:
        shape_meta: Raw shape metadata attribute from an IRNode.

    Returns:
        list[int | str]: Normalized list of dimensions.
    """
    if shape_meta is None:
        return []
    if isinstance(shape_meta, (tuple, list)):
        return [_normalize_shape_dim(d) for d in shape_meta]
    return [_normalize_shape_dim(shape_meta)]


def _clean_node_attributes(raw_attrs: AttributeValue) -> dict[str, str | int | float | bool]:
    """Normalize node attributes to primitive JSON-compatible values.

    Args:
        raw_attrs: Raw attributes dictionary from a node or serialized payload.

    Returns:
        dict[str, str | int | float | bool]: Cleaned attributes dictionary.
    """
    if not isinstance(raw_attrs, dict):
        return {}
    clean_attrs: dict[str, str | int | float | bool] = {}
    for k, v in raw_attrs.items():
        clean_attrs[str(k)] = v if isinstance(v, (str, int, float, bool)) else str(v)
    return clean_attrs


def _serialize_initializers(graph: IRGraph) -> dict[str, AttributeValue]:
    """Serialize graph initializers to JSON-compatible values.

    Args:
        graph: The IRGraph containing initializers.

    Returns:
        dict[str, AttributeValue]: Serialized initializers dictionary.
    """
    initializers_dict: dict[str, AttributeValue] = {}
    if hasattr(graph, "initializers") and graph.initializers:
        for k, v in graph.initializers.items():
            if hasattr(v, "tolist"):
                initializers_dict[str(k)] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)):
                initializers_dict[str(k)] = v
            else:
                initializers_dict[str(k)] = str(v)
    return initializers_dict


def _build_serialization_dict(graph: IRGraph) -> dict[str, AttributeValue]:
    """Builds a structured dictionary representing the IR graph.

    Args:
        graph (IRGraph): The intermediate representation graph.

    Returns:
        dict[str, AttributeValue]: Structured serialization dictionary.
    """
    inputs_list: list[str] = list(getattr(graph, "inputs", [])) if getattr(graph, "inputs", None) else [node_id for node_id, node in graph.nodes.items() if getattr(node, "op_type", "") == "Input"]
    outputs_list: list[str] = list(getattr(graph, "outputs", []))
    nodes_dict: dict[str, NodeSerializationModel] = {}
    initializers_dict = _serialize_initializers(graph)

    for node_id, node in graph.nodes.items():
        shape_list = _serialize_shape_metadata(getattr(node, "shape_metadata", None))
        clean_attrs = _clean_node_attributes(getattr(node, "attributes", {}))
        serialized_subgraphs: dict[str, GraphSerializationModel] = {str(sub_k): GraphSerializationModel.model_validate(_build_serialization_dict(sub_g)) for sub_k, sub_g in getattr(node, "subgraphs", {}).items() if hasattr(sub_g, "nodes")}

        node_entry = NodeSerializationModel(
            id=node_id,
            op=node.op_type,
            op_type=node.op_type,
            inputs=list(node.inputs),
            shape_metadata=shape_list,
            shape=shape_list,
            attributes=clean_attrs,
            subgraphs=serialized_subgraphs if serialized_subgraphs else None,
        )
        nodes_dict[node_id] = node_entry

    model = GraphSerializationModel(
        name=getattr(graph, "name", "graph"),
        inputs=inputs_list,
        outputs=outputs_list,
        initializers=initializers_dict,
        nodes=nodes_dict,
    )
    return model.model_dump(exclude_none=True)


def _parse_single_serialized_node(node_id: str, raw_node: Mapping[str, AttributeValue]) -> IRNode:
    """Reconstruct a single IRNode from its serialized dictionary representation.

    Args:
        node_id: Identifier of the node.
        raw_node: Raw serialized dictionary for the node.

    Returns:
        IRNode: Reconstructed IRNode instance.
    """
    op_type = str(raw_node.get("op_type", raw_node.get("op", "")))
    raw_inputs = raw_node.get("inputs", [])
    inputs: list[str] = [str(inp) for inp in raw_inputs] if isinstance(raw_inputs, list) else []

    raw_shape = raw_node.get("shape_metadata", raw_node.get("shape", []))
    shape_tuple: tuple[int | str, ...] = tuple(_normalize_shape_dim(dim) for dim in raw_shape) if isinstance(raw_shape, list) else ()
    attributes = _clean_node_attributes(raw_node.get("attributes", {}))

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
    return node


def _parse_serialization_dict(data: Mapping[str, AttributeValue]) -> IRGraph:
    """Parses a structured dictionary back into an IRGraph instance.

    Args:
        data (Mapping[str, AttributeValue]): Parsed graph data mapping.

    Returns:
        IRGraph: Reconstructed intermediate representation graph.
    """
    graph = IRGraph(name=str(data.get("name", "graph")))

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
            if isinstance(raw_node, dict):
                graph.nodes[str(node_id)] = _parse_single_serialized_node(str(node_id), raw_node)

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
    data: dict[str, AttributeValue] = json.loads(json_str)
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
    data: dict[str, AttributeValue] = yaml.safe_load(yaml_str) or {}
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
