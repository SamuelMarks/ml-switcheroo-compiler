"""Low-level Protocol Buffers wire encoding, decoding, and validation for ONNX."""

from __future__ import annotations

import struct
import typing
from collections.abc import Sequence

from ml_switcheroo_ir import IRNode

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_int_list,
    coerce_str,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module

T = typing.TypeVar("T")


def _node_inputs_list(node: BackendArg) -> list[str]:
    """Return string list of input IDs on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        list[str]: Node input IDs.
    """
    raw = adapt_backend_module(node).get_symbol("inputs", [])
    return [str(x) for x in raw] if isinstance(raw, (list, tuple)) else []


def _node_shape_tuple(node: BackendArg) -> tuple[int, ...]:
    """Return integer shape tuple from `shape_metadata` on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        tuple[int, ...]: Shape dimensions tuple.
    """
    raw = adapt_backend_module(node).get_symbol("shape_metadata", ())
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in coerce_int_list(raw))
    return ()


def _node_attrs_dict(node: BackendArg) -> dict[str, BackendArg]:
    """Return attributes dictionary on `node`.

    Args:
        node (BackendArg): Target node.

    Returns:
        dict[str, BackendArg]: Attributes mapping.
    """
    raw = adapt_backend_module(node).get_symbol("attributes", {})
    return {str(k): v for k, v in raw.items()} if isinstance(raw, dict) else {}


class TensorProtoType(typing.Protocol):
    """Protocol for ONNX TensorProto objects."""

    name: str


class ValueInfoProtoType(typing.Protocol):
    """Protocol for ONNX ValueInfoProto objects."""

    name: str


@typing.runtime_checkable
class NodeProtoType(typing.Protocol):
    """Protocol for ONNX NodeProto objects."""

    name: str


@typing.runtime_checkable
class GraphProtoType(typing.Protocol):
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
        nid: str = coerce_str(adapt_backend_module(node).get_symbol("id", ""))
        op_type: str = coerce_str(adapt_backend_module(node).get_symbol("op_type", ""))
        if op_type == "Input":
            defined.add(nid)
            continue
        for inp in _node_inputs_list(node):
            if inp not in defined:
                raise ValueError(
                    f"Graph nodes are not topologically sorted: input '{inp}' is referenced before definition."
                )
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


def _parse_opset_import_version(binary_data: bytes, offset: int, sub_end: int) -> int:
    """Parse opset version from an OperatorSetIdProto submessage.

    Args:
        binary_data (bytes): Full ONNX binary buffer.
        offset (int): Start offset of the OperatorSetIdProto submessage.
        sub_end (int): End offset of the submessage.

    Returns:
        int: Extracted opset version or 0.
    """
    opset_version: int = 0
    while offset < sub_end:
        sub_tag, offset = decode_varint(binary_data, offset)
        sub_fn: int = sub_tag >> 3
        sub_wt: int = sub_tag & 0x07
        if sub_fn == 2 and sub_wt == 0:
            opset_version, offset = decode_varint(binary_data, offset)
        elif sub_wt == 0:
            _, offset = decode_varint(binary_data, offset)
        else:
            sub_len, offset = decode_varint(binary_data, offset)
            offset += sub_len
    return opset_version


def _skip_protobuf_wire_field(binary_data: bytes, offset: int, wire_type: int) -> int:
    """Advance offset past an unhandled protobuf field based on its wire type.

    Args:
        binary_data (bytes): Full protobuf binary buffer.
        offset (int): Current byte offset after reading the tag.
        wire_type (int): Protobuf wire type (0, 1, 2, or 5).

    Returns:
        int: New byte offset after skipping the field payload.

    Raises:
        ValueError: If wire_type is unknown.
    """
    if wire_type == 0:
        _, next_offset = decode_varint(binary_data, offset)
        return next_offset
    if wire_type == 2:
        length, next_offset = decode_varint(binary_data, offset)
        return next_offset + length
    if wire_type == 5:
        return offset + 4
    if wire_type == 1:
        return offset + 8
    raise ValueError(f"Malformed protobuf wire type {wire_type} at offset {offset}.")


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
            opset_version = _parse_opset_import_version(binary_data, offset, offset + length)
            offset += length
        else:
            offset = _skip_protobuf_wire_field(binary_data, offset, wire_type)

    if not has_graph:
        raise ValueError("ONNX ModelProto is missing a valid GraphProto payload.")

    return {
        "ir_version": ir_version,
        "producer_name": producer_name,
        "opset_version": opset_version,
    }
