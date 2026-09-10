# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""ONNX Target Emission and Pure-Python Binary Protobuf Serialization."""

from __future__ import annotations

import math
import os
import struct
import uuid
from collections.abc import Callable, Sequence
from typing import Optional, Protocol, TypeVar, Union

import yaml

from ml_switcheroo_compiler.backends.base_generator import BaseGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode

T = TypeVar("T")


class TensorProtoType(Protocol):
    """Protocol for ONNX TensorProto objects."""

    name: str


class ValueInfoProtoType(Protocol):
    """Protocol for ONNX ValueInfoProto objects."""

    name: str


class NodeProtoType(Protocol):
    """Protocol for ONNX NodeProto objects."""

    name: str


class GraphProtoType(Protocol):
    """Protocol for ONNX GraphProto objects."""

    name: str


def encode_varint(value: int) -> bytes:
    """Encode an unsigned integer into Protocol Buffers varint bytes.

    Args:
        value (int): Non-negative integer to encode.

    Returns:
        bytes: Encoded varint byte sequence.
    """
    out: bytearray = bytearray()
    while value > 0x7F:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    out.append(value & 0x7F)
    return bytes(out)


def encode_tag(field_number: int, wire_type: int) -> bytes:
    """Encode field number and wire type into protobuf tag varint.

    Args:
        field_number (int): Protobuf field tag number.
        wire_type (int): Wire type code (0: varint, 1: 64-bit, 2: length-delimited, 5: 32-bit).

    Returns:
        bytes: Encoded tag bytes.
    """
    return encode_varint((field_number << 3) | wire_type)


def encode_length_delimited(field_number: int, data: bytes) -> bytes:
    """Encode length-delimited wire type (2) protobuf field.

    Args:
        field_number (int): Protobuf field tag number.
        data (bytes): Byte payload to delimit.

    Returns:
        bytes: Tag + length varint + raw payload bytes.
    """
    return encode_tag(field_number, 2) + encode_varint(len(data)) + data


def encode_string(field_number: int, text: str) -> bytes:
    """Encode UTF-8 string protobuf field.

    Args:
        field_number (int): Protobuf field tag number.
        text (str): String content.

    Returns:
        bytes: Encoded length-delimited string field.
    """
    return encode_length_delimited(field_number, text.encode("utf-8"))


def encode_int(field_number: int, val: int) -> bytes:
    """Encode signed/unsigned 64-bit integer protobuf field.

    Args:
        field_number (int): Protobuf field tag number.
        val (int): Integer value.

    Returns:
        bytes: Encoded varint field.
    """
    i_val: int = val
    if i_val < 0:
        i_val += 1 << 64
    return encode_tag(field_number, 0) + encode_varint(i_val)


def encode_float(field_number: int, val: float) -> bytes:
    """Encode 32-bit little-endian float protobuf field.

    Args:
        field_number (int): Protobuf field tag number.
        val (float): Floating-point value.

    Returns:
        bytes: Tag (wire 5) + 4-byte packed float.
    """
    return encode_tag(field_number, 5) + struct.pack("<f", float(val))


def serialize_dimension(dim_val: int | str | None) -> bytes:
    """Serialize a single Dimension into TensorShapeProto.Dimension bytes.

    Args:
        dim_val (Union[int, str, None]): Dimension value (fixed integer or symbolic string).

    Returns:
        bytes: Wire bytes for Dimension message.
    """
    if dim_val is None or (isinstance(dim_val, str) and not dim_val.isdigit()) or (isinstance(dim_val, int) and dim_val < 0):
        param_name: str = str(dim_val) if dim_val is not None and str(dim_val) != "-1" else "batch_size"
        return encode_string(2, param_name)
    return encode_int(1, int(dim_val))


def serialize_tensor_shape(shape: Sequence[int | str | None]) -> bytes:
    """Serialize tensor shape dimensions into TensorShapeProto bytes.

    Args:
        shape (Sequence[Union[int, str, None]]): Tuple or list of dimensions.

    Returns:
        bytes: Wire bytes for TensorShapeProto message.
    """
    dims_bytes: list[bytes] = [encode_length_delimited(1, serialize_dimension(d)) for d in shape]
    return b"".join(dims_bytes)


def serialize_type_proto(elem_type: int, shape: Sequence[int | str | None]) -> bytes:
    """Serialize element type and shape into TypeProto bytes.

    Args:
        elem_type (int): ONNX TensorProto data type code.
        shape (Sequence[Union[int, str, None]]): Tensor dimensions.

    Returns:
        bytes: Wire bytes for TypeProto message.
    """
    shape_bytes: bytes = serialize_tensor_shape(shape)
    tensor_type_bytes: bytes = encode_int(1, elem_type) + encode_length_delimited(2, shape_bytes)
    return encode_length_delimited(1, tensor_type_bytes)


def serialize_value_info(name: str, elem_type: int, shape: Sequence[int | str | None]) -> bytes:
    """Serialize input or output tensor metadata into ValueInfoProto bytes.

    Args:
        name (str): Identifier name of the tensor.
        elem_type (int): ONNX TensorProto primitive data type code.
        shape (Sequence[Union[int, str, None]]): Dimensions sequence.

    Returns:
        bytes: Wire bytes for ValueInfoProto message.
    """
    type_proto_bytes: bytes = serialize_type_proto(elem_type, shape)
    return encode_string(1, name) + encode_length_delimited(2, type_proto_bytes)


def serialize_tensor_proto(name: str, data_type: int, dims: Sequence[int], raw_bytes: bytes) -> bytes:
    """Serialize raw tensor data into TensorProto bytes.

    Args:
        name (str): Identifier name.
        data_type (int): TensorProto data type enum.
        dims (Sequence[int]): Shape dimensions.
        raw_bytes (bytes): Packed raw array bytes.

    Returns:
        bytes: Wire bytes for TensorProto message.
    """
    dims_bytes: bytes = b"".join(encode_int(1, int(d)) for d in dims)
    return dims_bytes + encode_int(2, data_type) + encode_string(8, name) + encode_length_delimited(9, raw_bytes)


def serialize_attribute(name: str, val: object, graph_serializer: Callable[[IRGraph, str], bytes] | None = None) -> bytes:
    """Serialize a single node attribute into AttributeProto bytes.

    Args:
        name (str): Attribute key name.
        val (object): Attribute payload value.
        graph_serializer (Optional[Callable[[IRGraph, str], bytes]]): Subgraph serializer callback.

    Returns:
        bytes: Wire bytes for AttributeProto message.
    """
    attr_bytes: bytes = encode_string(1, name)
    if isinstance(val, float):
        return attr_bytes + encode_float(2, val) + encode_int(20, 1)  # FLOAT
    if isinstance(val, int) and not isinstance(val, bool):
        return attr_bytes + encode_int(3, val) + encode_int(20, 2)  # INT
    if isinstance(val, str):
        return attr_bytes + encode_string(4, val) + encode_int(20, 3)  # STRING
    if isinstance(val, (bytes, bytearray)):
        return attr_bytes + encode_length_delimited(4, bytes(val)) + encode_int(20, 3)  # STRING
    if isinstance(val, (list, tuple)):
        if val and isinstance(val[0], int):
            ints_bytes: bytes = b"".join(encode_int(8, int(x)) for x in val)
            return attr_bytes + ints_bytes + encode_int(20, 7)  # INTS
        if val and isinstance(val[0], float):
            floats_bytes: bytes = b"".join(encode_tag(7, 5) + struct.pack("<f", float(x)) for x in val)
            return attr_bytes + floats_bytes + encode_int(20, 6)  # FLOATS
        if val and isinstance(val[0], str):
            strs_bytes: bytes = b"".join(encode_length_delimited(9, str(x).encode("utf-8")) for x in val)
            return attr_bytes + strs_bytes + encode_int(20, 8)  # STRINGS
        return attr_bytes + encode_int(20, 7)
    if hasattr(val, "nodes") and graph_serializer is not None:
        subgraph_bytes: bytes = graph_serializer(val, f"{name}_subgraph")
        return attr_bytes + encode_length_delimited(6, subgraph_bytes) + encode_int(20, 5)  # GRAPH
    return attr_bytes + encode_string(4, str(val)) + encode_int(20, 3)


def serialize_node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str,
    attributes: Sequence[bytes] = (),
    domain: str = "",
) -> bytes:
    """Serialize an operation node into NodeProto bytes.

    Args:
        op_type (str): ONNX operator name.
        inputs (Sequence[str]): Input variable names.
        outputs (Sequence[str]): Output variable names.
        name (str): Unique node name.
        attributes (Sequence[bytes]): Pre-serialized attribute bytes.
        domain (str): Operator domain string.

    Returns:
        bytes: Wire bytes for NodeProto message.
    """
    in_bytes: bytes = b"".join(encode_string(1, inp) for inp in inputs)
    out_bytes: bytes = b"".join(encode_string(2, out) for out in outputs)
    name_bytes: bytes = encode_string(3, name)
    op_bytes: bytes = encode_string(4, op_type)
    attr_bytes: bytes = b"".join(encode_length_delimited(5, a) for a in attributes)
    domain_bytes: bytes = encode_string(7, domain) if domain else b""
    return in_bytes + out_bytes + name_bytes + op_bytes + attr_bytes + domain_bytes


def serialize_graph(
    name: str,
    nodes: Sequence[bytes],
    inputs: Sequence[bytes],
    outputs: Sequence[bytes],
    initializers: Sequence[bytes] = (),
) -> bytes:
    """Serialize full computation graph into GraphProto bytes.

    Args:
        name (str): Graph name identifier.
        nodes (Sequence[bytes]): Serialized NodeProto messages.
        inputs (Sequence[bytes]): Serialized input ValueInfoProto messages.
        outputs (Sequence[bytes]): Serialized output ValueInfoProto messages.
        initializers (Sequence[bytes]): Serialized initializer TensorProto messages.

    Returns:
        bytes: Wire bytes for GraphProto message.
    """
    nodes_bytes: bytes = b"".join(encode_length_delimited(1, n) for n in nodes)
    name_bytes: bytes = encode_string(2, name)
    inits_bytes: bytes = b"".join(encode_length_delimited(5, init) for init in initializers)
    inputs_bytes: bytes = b"".join(encode_length_delimited(11, i) for i in inputs)
    outputs_bytes: bytes = b"".join(encode_length_delimited(12, o) for o in outputs)
    return nodes_bytes + name_bytes + inits_bytes + inputs_bytes + outputs_bytes


def serialize_operator_set_id(domain: str = "", version: int = 18) -> bytes:
    """Serialize operator set version into OperatorSetIdProto bytes.

    Args:
        domain (str): Operator set domain identifier.
        version (int): Opset version number.

    Returns:
        bytes: Wire bytes for OperatorSetIdProto message.
    """
    domain_bytes: bytes = encode_string(1, domain) if domain else b""
    version_bytes: bytes = encode_int(2, version)
    return domain_bytes + version_bytes


def serialize_model(
    graph_bytes: bytes,
    producer_name: str = "ml-switcheroo-compiler",
    opset_version: int = 18,
    ir_version: int = 8,
) -> bytes:
    """Serialize root model wrapper into ModelProto bytes.

    Args:
        graph_bytes (bytes): Pre-serialized GraphProto payload.
        producer_name (str): Author identification string.
        opset_version (int): Target ONNX opset version.
        ir_version (int): ONNX intermediate representation format version.

    Returns:
        bytes: Full binary protobuf wire bytes representing an executable .onnx model.
    """
    ir_bytes: bytes = encode_int(1, ir_version)
    producer_bytes: bytes = encode_string(2, producer_name)
    version_bytes: bytes = encode_string(3, "0.1.0")
    g_bytes: bytes = encode_length_delimited(7, graph_bytes)
    opset_bytes: bytes = encode_length_delimited(8, serialize_operator_set_id("", opset_version))
    return ir_bytes + producer_bytes + version_bytes + g_bytes + opset_bytes


def validate_topological_sort(nodes: Sequence[IRNode], graph_inputs: Sequence[str]) -> bool:
    """Validate that IR nodes are topologically sorted.

    Args:
        nodes (Sequence[IRNode]): Sequence of IR nodes in execution order.
        graph_inputs (Sequence[str]): Input identifiers of the computation graph.

    Returns:
        bool: True if topologically sorted.

    Raises:
        ValueError: If any node input is referenced before its definition.
    """
    defined: set[str] = set(graph_inputs)
    for node in nodes:
        nid: str = getattr(node, "id", "")
        op_type: str = getattr(node, "op_type", "")
        if op_type == "Input":
            defined.add(nid)
            continue
        for inp in getattr(node, "inputs", []):
            if inp not in defined:
                raise ValueError(f"Graph nodes are not topologically sorted: input '{inp}' is referenced before definition.")
        defined.add(nid)
    return True


def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode a varint from raw bytes starting at offset.

    Args:
        data (bytes): Byte buffer.
        offset (int): Starting offset.

    Returns:
        tuple[int, int]: Decoded integer and new offset.
    """
    res: int = 0
    shift: int = 0
    while offset < len(data):
        b: int = data[offset]
        offset += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, offset


def validate_onnx_model_bytes(binary_data: bytes) -> dict[str, int | str]:
    """Validate raw ONNX ModelProto binary bytes and extract metadata without external dependencies.

    Args:
        binary_data (bytes): Serialized ONNX model bytes.

    Returns:
        dict[str, int | str]: Parsed model metadata including ir_version, producer_name, and opset_version.

    Raises:
        ValueError: If binary data is empty or structurally invalid protobuf.
    """
    if not binary_data or len(binary_data) < 4:
        raise ValueError("ONNX binary payload is empty or too short.")

    offset: int = 0
    ir_version: int = 0
    producer_name: str = ""
    opset_version: int = 0
    has_graph: bool = False

    while offset < len(binary_data):
        tag_val, offset = decode_varint(binary_data, offset)
        field_number: int = tag_val >> 3
        wire_type: int = tag_val & 0x07

        if field_number == 1 and wire_type == 0:
            ir_version, offset = decode_varint(binary_data, offset)
        elif field_number == 2 and wire_type == 2:
            length, offset = decode_varint(binary_data, offset)
            producer_name = binary_data[offset : offset + length].decode("utf-8", errors="replace")
            offset += length
        elif field_number == 7 and wire_type == 2:
            length, offset = decode_varint(binary_data, offset)
            has_graph = length > 0
            offset += length
        elif field_number == 8 and wire_type == 2:
            length, offset = decode_varint(binary_data, offset)
            sub_end: int = offset + length
            while offset < sub_end:
                sub_tag, offset = decode_varint(binary_data, offset)
                sub_fn: int = sub_tag >> 3
                sub_wt: int = sub_tag & 0x07
                if sub_fn == 2 and sub_wt == 0:
                    opset_version, offset = decode_varint(binary_data, offset)
                elif sub_wt == 0:
                    _, offset = decode_varint(binary_data, offset)
                elif sub_wt == 2:
                    sub_len, offset = decode_varint(binary_data, offset)
                    offset += sub_len
                else:
                    offset = sub_end
                    break
            offset = sub_end
        elif wire_type == 0:
            _, offset = decode_varint(binary_data, offset)
        elif wire_type == 2:
            length, offset = decode_varint(binary_data, offset)
            offset += length
        elif wire_type == 5:
            offset += 4
        elif wire_type == 1:
            offset += 8
        else:
            raise ValueError(f"Malformed protobuf wire type {wire_type} at offset {offset}.")

    if not has_graph:
        raise ValueError("ONNX ModelProto is missing a valid GraphProto payload.")

    return {
        "ir_version": ir_version,
        "producer_name": producer_name,
        "opset_version": opset_version,
    }


class ONNXCodeGenerator(BaseGenerator):
    """ONNX Code Generator for emitting ONNX graph representations and binary protobuf files.

    Attributes:
        graph (IRGraph): The IR graph to process.
        var_map (dict[str, str]): Mapping of IR node IDs to generated variable names.
    """

    def __init__(self, graph: IRGraph, delegates: T | None = None) -> None:
        """Initialize ONNXCodeGenerator.

        Args:
            graph (IRGraph): The IR graph to process.
            delegates (Optional[T], optional): Visitor delegates.
        """
        super().__init__(graph, delegates)
        self.var_map: dict[str, str] = {}

        yaml_path: str = os.path.join(os.path.dirname(__file__), "onnx_schema.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                self.schema = yaml.safe_load(f)
        else:
            self.schema = {}

    def generic_visit(self, node: IRNode, input_vars: list[str], **kwargs: str) -> str:
        """Process a node and return its generated ONNX variable name.

        Args:
            node (IRNode): The IR node.
            input_vars (list[str]): Names of the input variables.
            **kwargs (str): Additional attributes.

        Returns:
            str: Variable name of the evaluated node.
        """
        if node is None:
            return "onnx_op"

        nid: str = getattr(node, "id", str(uuid.uuid4()))
        self.var_map[nid] = nid
        return nid

    def _get_proto_type(self, dt: str, TensorProto: type[TensorProtoType] | None = None) -> int:
        """Map data type string to ONNX TensorProto primitive integer code.

        Args:
            dt (str): The dtype string.
            TensorProto (Optional[type[TensorProtoType]]): Optional ONNX TensorProto namespace.

        Returns:
            int: The protobuf enum type.
        """
        dt_str: str = str(dt).lower()
        dt_map: dict[str, int] = self.schema.get("types", {})
        return dt_map.get(dt_str, 1)

    def _generate_text_fallback(self) -> str:
        """Generate a text-proto fallback string representation in case ONNX is not available.

        Returns:
            str: Serialized ONNX text representation.
        """
        lines: list[str] = ["ir_version: 7", 'producer_name: "ml-switcheroo-compiler"', "graph {"]

        for node in self.sorted_nodes:
            if getattr(node, "op_type", "") == "Input":
                nid: str = getattr(node, "id", "")
                shape: str = "x".join(str(s) for s in (getattr(node, "shape_metadata", ()) or ()))
                dtype: str = getattr(node, "dtype", "float32")
                lines.append(f'  input: "{nid}" [shape: {shape}, dtype: {dtype}]')

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type != "Input":
                nid: str = getattr(node, "id", "")
                inps: str = ", ".join(f'"{i}"' for i in getattr(node, "inputs", []))
                lines.append(f'  "{nid}" = {op_type}({inps})')

        for out_id in getattr(self.graph, "outputs", []) or []:
            lines.append(f'  output: "{out_id}"')

        lines.append("}")
        return "\n".join(lines)

    def _get_node_and_name(self, item: IRNode | str, is_output: bool) -> tuple[IRNode | None, str]:
        """Retrieve a node object and its ID name.

        Args:
            item (Union[IRNode, str]): The IR node or its ID string.
            is_output (bool): True if looking up an output node by ID.

        Returns:
            tuple[Optional[IRNode], str]: A tuple containing the node and its name string.
        """
        if is_output:
            out_id: str = str(item)
            node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == out_id), None)
            return node, out_id
        if isinstance(item, IRNode):
            return item, str(getattr(item, "id", ""))
        node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == str(item)), None)
        return node, str(item)

    def _build_single_value_info(
        self,
        item: IRNode | str,
        dynamic_axes: dict[str, dict[int, str]] | None,
        TensorProto: type[TensorProtoType],
        is_output: bool,
    ) -> ValueInfoProtoType:
        """Construct an ONNX TensorValueInfoProto for a single node.

        Args:
            item (Union[IRNode, str]): The IR node or output ID.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for an output.

        Returns:
            ValueInfoProtoType: An ONNX ValueInfoProto object.
        """
        from onnx import helper

        node, name = self._get_node_and_name(item, is_output)
        shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or () if node else ()
        dt: str = getattr(node, "dtype", "float32") if node else "float32"
        proto_type: int = self._get_proto_type(dt, TensorProto)
        shape_list: list[int | str] = list(shape)

        if dynamic_axes and name in dynamic_axes:
            for axis_idx, axis_name in dynamic_axes[name].items():
                shape_list[axis_idx] = axis_name
        return helper.make_tensor_value_info(name, proto_type, shape_list)

    def _build_onnx_value_infos(
        self,
        nodes_or_ids: list[IRNode | str],
        dynamic_axes: dict[str, dict[int, str]] | None,
        TensorProto: type[TensorProtoType],
        is_output: bool = False,
    ) -> list[ValueInfoProtoType]:
        """Construct a list of ONNX TensorValueInfoProtos.

        Args:
            nodes_or_ids (list[Union[IRNode, str]]): List of IR nodes or output IDs.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.
            is_output (bool): True if building for outputs.

        Returns:
            list[ValueInfoProtoType]: A list of ONNX ValueInfoProto objects.
        """
        return [self._build_single_value_info(item, dynamic_axes, TensorProto, is_output) for item in nodes_or_ids]

    def _build_onnx_nodes(self, TensorProto: type[TensorProtoType]) -> list[NodeProtoType]:
        """Construct all intermediate ONNX NodeProtos for the graph.

        Args:
            TensorProto (type[TensorProtoType]): The ONNX TensorProto namespace object.

        Returns:
            list[NodeProtoType]: A list of ONNX NodeProto objects.
        """
        from onnx import helper

        onnx_nodes: list[NodeProtoType] = []
        from ml_switcheroo_compiler.ops.registry import _YAML_REGISTRY as OPS_REGISTRY

        def get_onnx_op_name(op_type: str) -> str:
            """Map op_type to target ONNX opcode."""
            op_def = OPS_REGISTRY.get(op_type, {})
            variants = op_def.get("variants", {})
            if "edge_onnx" in variants:
                gen: str = variants["edge_onnx"].get("generator")
                if gen:
                    return gen
            if op_type not in self.schema.get("operations", {}):
                from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

                raise BackendNotSupportedError(f"Operation '{op_type}' not supported in ONNX schema.")
            return str(op_type)

        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid: str = getattr(node, "id", "")
            inputs: list[str] = getattr(node, "inputs", [])

            if op_type == "Constant":
                val: float = node.attributes.get("value", 0.0)
                dt: str = getattr(node, "dtype", "float32")
                shape: tuple[int, ...] = getattr(node, "shape_metadata", ()) or ()
                proto_type: int = self._get_proto_type(dt, TensorProto)
                num_elements: int = math.prod(shape) if shape else 1
                tensor_proto = helper.make_tensor(
                    name=nid,
                    data_type=proto_type,
                    dims=list(shape),
                    vals=[val] * num_elements,
                )
                onnx_nodes.append(helper.make_node("Constant", inputs=[], outputs=[nid], name=nid, value=tensor_proto))
            else:
                onnx_op: str = get_onnx_op_name(op_type)
                kwargs: dict[str, str | int | float | list[int] | GraphProtoType] = {}
                if op_type == "If":
                    if "then_branch" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["then_branch"])
                        kwargs["then_branch"] = subgen._build_onnx_graph(None)
                        kwargs["then_branch"].name = f"{nid}_then"
                    if "else_branch" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["else_branch"])
                        kwargs["else_branch"] = subgen._build_onnx_graph(None)
                        kwargs["else_branch"].name = f"{nid}_else"
                elif op_type in ("Loop", "WhileLoop"):
                    if "body" in node.attributes:
                        subgen = ONNXCodeGenerator(node.attributes["body"])
                        kwargs["body"] = subgen._build_onnx_graph(None)
                        kwargs["body"].name = f"{nid}_body"

                onnx_nodes.append(helper.make_node(onnx_op, inputs=inputs, outputs=[nid], name=nid, **kwargs))
        return onnx_nodes

    def _build_onnx_graph(self, dynamic_axes: dict[str, dict[int, str]] | None = None) -> GraphProtoType:
        """Construct the full ONNX GraphProto.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis mapping configuration.

        Returns:
            GraphProtoType: The ONNX GraphProto object.
        """
        from onnx import TensorProto, helper

        input_nodes = [n for n in self.sorted_nodes if getattr(n, "op_type", "") == "Input"]
        onnx_inputs = self._build_onnx_value_infos(input_nodes, dynamic_axes, TensorProto, is_output=False)

        output_ids: list[str] = getattr(self.graph, "outputs", []) or []
        onnx_outputs = self._build_onnx_value_infos(output_ids, dynamic_axes, TensorProto, is_output=True)

        onnx_nodes = self._build_onnx_nodes(TensorProto)
        return helper.make_graph(onnx_nodes, "ml_switcheroo_graph", onnx_inputs, onnx_outputs)

    def serialize_model_to_bytes(
        self,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 18,
    ) -> bytes:
        """Serialize the IR Graph into strictly compliant pure-Python ONNX binary protobuf bytes.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): Dynamic axis configuration.
            opset_version (int): Target ONNX opset version (14 to 20).

        Returns:
            bytes: Valid ONNX binary protobuf byte sequence.

        Raises:
            ValueError: If opset_version is unsupported or topological sort fails.
        """
        if not (14 <= opset_version <= 20):
            raise ValueError(f"Unsupported ONNX opset_version {opset_version}: must be between 14 and 20.")

        validate_topological_sort(self.sorted_nodes, getattr(self.graph, "inputs", []) or [])

        # Build inputs
        inputs_bytes: list[bytes] = []
        for n in self.sorted_nodes:
            if getattr(n, "op_type", "") == "Input":
                name: str = getattr(n, "id", "")
                dt: str = getattr(n, "dtype", "float32")
                shape: list[int | str | None] = list(getattr(n, "shape_metadata", ()) or ())
                if dynamic_axes and name in dynamic_axes:
                    for axis_idx, axis_name in dynamic_axes[name].items():
                        if axis_idx < len(shape):
                            shape[axis_idx] = axis_name
                p_type: int = self._get_proto_type(dt)
                inputs_bytes.append(serialize_value_info(name, p_type, shape))

        # Build outputs
        outputs_bytes: list[bytes] = []
        for out_id in getattr(self.graph, "outputs", []) or []:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == str(out_id)), None)
            dt = getattr(out_node, "dtype", "float32") if out_node else "float32"
            shape = list(getattr(out_node, "shape_metadata", ()) or ()) if out_node else []
            if dynamic_axes and str(out_id) in dynamic_axes:
                for axis_idx, axis_name in dynamic_axes[str(out_id)].items():
                    if axis_idx < len(shape):
                        shape[axis_idx] = axis_name
            p_type = self._get_proto_type(dt)
            outputs_bytes.append(serialize_value_info(str(out_id), p_type, shape))

        # Subgraph serializer helper
        def sub_serializer(subgraph: IRGraph, sub_name: str) -> bytes:
            """Serialize nested subgraph to ONNX GraphProto bytes.

            Args:
                subgraph (IRGraph): The sub-graph to serialize.
                sub_name (str): Identifier name for the sub-graph.

            Returns:
                bytes: Serialized GraphProto.
            """
            subgen = ONNXCodeGenerator(subgraph)
            return subgen._serialize_graph_internal(sub_name, None)

        # Build nodes
        nodes_bytes: list[bytes] = []
        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue

            nid: str = getattr(node, "id", "")
            inputs: list[str] = list(getattr(node, "inputs", []))
            outputs: list[str] = [nid]
            attributes: list[bytes] = []

            if op_type == "Constant":
                c_val = getattr(node, "attributes", {}).get("value", 0.0)
                dt = getattr(node, "dtype", "float32")
                shape_c = [int(x) for x in (getattr(node, "shape_metadata", ()) or ())]
                p_type = self._get_proto_type(dt)
                num_elem: int = math.prod(shape_c) if shape_c else 1
                raw_data: bytes = struct.pack(f"<{num_elem}f", *([float(c_val)] * num_elem))
                t_proto: bytes = serialize_tensor_proto(nid, p_type, shape_c, raw_data)
                # Attribute 'value'
                attr_c: bytes = encode_string(1, "value") + encode_length_delimited(5, t_proto) + encode_int(20, 4)
                attributes.append(attr_c)
                nodes_bytes.append(serialize_node("Constant", [], outputs, nid, attributes))
            elif op_type in ("If", "Cond"):
                then_b = getattr(node, "attributes", {}).get("then_branch")
                else_b = getattr(node, "attributes", {}).get("else_branch")
                if then_b is not None:
                    attributes.append(serialize_attribute("then_branch", then_b, sub_serializer))
                if else_b is not None:
                    attributes.append(serialize_attribute("else_branch", else_b, sub_serializer))
                nodes_bytes.append(serialize_node("If", inputs, outputs, nid, attributes))
            elif op_type in ("Loop", "WhileLoop"):
                body_b = getattr(node, "attributes", {}).get("body")
                if body_b is not None:
                    attributes.append(serialize_attribute("body", body_b, sub_serializer))
                nodes_bytes.append(serialize_node("Loop", inputs, outputs, nid, attributes))
            else:
                for k, v in getattr(node, "attributes", {}).items():
                    if k in ("dtype", "shape", "is_state", "is_state_update", "param_name", "variable_name", "name"):
                        continue
                    attributes.append(serialize_attribute(k, v, sub_serializer))
                nodes_bytes.append(serialize_node(op_type, inputs, outputs, nid, attributes))

        graph_payload: bytes = serialize_graph("ml_switcheroo_graph", nodes_bytes, inputs_bytes, outputs_bytes)
        return serialize_model(graph_payload, producer_name="ml-switcheroo-compiler", opset_version=opset_version, ir_version=8)

    def _serialize_graph_internal(self, graph_name: str, dynamic_axes: dict[str, dict[int, str]] | None) -> bytes:
        """Internal helper to serialize a subgraph into GraphProto bytes."""
        inputs_bytes: list[bytes] = []
        for n in self.sorted_nodes:
            if getattr(n, "op_type", "") == "Input":
                name: str = getattr(n, "id", "")
                dt: str = getattr(n, "dtype", "float32")
                shape: list[int | str | None] = list(getattr(n, "shape_metadata", ()) or ())
                p_type: int = self._get_proto_type(dt)
                inputs_bytes.append(serialize_value_info(name, p_type, shape))

        outputs_bytes: list[bytes] = []
        for out_id in getattr(self.graph, "outputs", []) or []:
            out_node = next((n for n in self.sorted_nodes if getattr(n, "id", None) == str(out_id)), None)
            dt = getattr(out_node, "dtype", "float32") if out_node else "float32"
            shape = list(getattr(out_node, "shape_metadata", ()) or ()) if out_node else []
            p_type = self._get_proto_type(dt)
            outputs_bytes.append(serialize_value_info(str(out_id), p_type, shape))

        nodes_bytes: list[bytes] = []
        for node in self.sorted_nodes:
            op_type: str = getattr(node, "op_type", "")
            if op_type == "Input":
                continue
            nid: str = getattr(node, "id", "")
            inputs: list[str] = list(getattr(node, "inputs", []))
            outputs: list[str] = [nid]
            attributes: list[bytes] = []
            for k, v in getattr(node, "attributes", {}).items():
                if k in ("dtype", "shape", "is_state", "is_state_update", "param_name", "variable_name", "name"):
                    continue
                attributes.append(serialize_attribute(k, v))
            nodes_bytes.append(serialize_node(op_type, inputs, outputs, nid, attributes))

        return serialize_graph(graph_name, nodes_bytes, inputs_bytes, outputs_bytes)

    def generate(self, dynamic_axes: dict[str, dict[int, str]] | None = None) -> str:
        """Generate a readable string/text-proto representation of the ONNX Graph.

        Args:
            dynamic_axes (Optional[dict[str, dict[int, str]]]): The dynamic_axes parameter.

        Returns:
            str: Result.
        """
        try:
            from onnx import helper

            graph_def = self._build_onnx_graph(dynamic_axes)
            try:
                from onnx import printer

                res: str = printer.to_text(graph_def)
                if not isinstance(res, str):
                    return str(res)
                return res
            except (ImportError, AttributeError):
                return str(helper.printable_graph(graph_def))
        except (ImportError, AttributeError):
            return self._generate_text_fallback()

    def export_onnx(
        self,
        file_path: str,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 18,
    ) -> None:
        """Export the IR Graph as a real, compliant binary .onnx file to disk.

        Args:
            file_path (str): The file_path parameter.
            dynamic_axes (Optional[dict[str, dict[int, str]]]): The dynamic_axes parameter.
            opset_version (int): Target ONNX opset version (14 to 20).
        """
        binary_data: bytes = self.serialize_model_to_bytes(dynamic_axes, opset_version=opset_version)

        try:
            from onnx import checker, load_model_from_string

            model_def = load_model_from_string(binary_data)
            checker.check_model(model_def)
        except (ImportError, AttributeError):
            validate_onnx_model_bytes(binary_data)

        with open(file_path, "wb") as f:
            f.write(binary_data)

    def verify(self, file_path: str | None = None) -> bool:
        """Verify the serialized ONNX model binary structure.

        Args:
            file_path (Optional[str]): Optional path to saved .onnx file; if None, in-memory model is verified.

        Returns:
            bool: True if model is verified valid.
        """
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data: bytes = f.read()
        else:
            data = self.serialize_model_to_bytes()
        meta: dict[str, int | str] = validate_onnx_model_bytes(data)
        return bool(int(meta.get("opset_version", 0)) >= 14 or int(meta.get("ir_version", 0)) > 0)
