"""Pure-Python ONNX Protobuf message serialization functions."""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence

from ml_switcheroo_ir import IRGraph

from ml_switcheroo_compiler.backends.eager_registry import (
    BackendArg,
    coerce_float,
    coerce_int,
)
from ml_switcheroo_compiler.backends.edge._onnx_wire import (
    encode_float,
    encode_int,
    encode_length_delimited,
    encode_string,
    encode_tag,
)
from ml_switcheroo_compiler.backends.protocols import adapt_backend_module
from ml_switcheroo_compiler.ir.core import ensure_graph


def serialize_dimension(dim_val: int | str | None) -> bytes:
    """Serialize a single Dimension into TensorShapeProto.Dimension bytes.

    Args:
        dim_val (int | str | None): Dimension value (fixed integer or symbolic string).

    Returns:
        bytes: Wire bytes for Dimension message.
    """
    if (
        dim_val is None
        or (isinstance(dim_val, str) and not dim_val.isdigit())
        or (isinstance(dim_val, int) and dim_val < 0)
    ):
        param_name: str = str(dim_val) if dim_val is not None and str(dim_val) != "-1" else "batch_size"
        return encode_string(2, param_name)
    return encode_int(1, int(dim_val))


def serialize_tensor_shape(shape: Sequence[int | str | None]) -> bytes:
    """Serialize tensor shape dimensions into TensorShapeProto bytes.

    Args:
        shape (Sequence[int | str | None]): Tuple or list of dimensions.

    Returns:
        bytes: Wire bytes for TensorShapeProto message.
    """
    dims_bytes: list[bytes] = [encode_length_delimited(1, serialize_dimension(d)) for d in shape]
    return b"".join(dims_bytes)


def serialize_type_proto(elem_type: int, shape: Sequence[int | str | None]) -> bytes:
    """Serialize element type and shape into TypeProto bytes.

    Args:
        elem_type (int): ONNX TensorProto data type code.
        shape (Sequence[int | str | None]): Tensor dimensions.

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
        shape (Sequence[int | str | None]): Dimensions sequence.

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


def _serialize_sequence_attribute(attr_bytes: bytes, val: Sequence[BackendArg]) -> bytes:
    """Serialize a sequence attribute (INTS, FLOATS, STRINGS) into AttributeProto bytes.

    Args:
        attr_bytes (bytes): Encoded attribute name prefix.
        val (Sequence[BackendArg]): Sequence of attribute elements.

    Returns:
        bytes: Wire bytes for sequence AttributeProto message.
    """
    if val and isinstance(val[0], int):
        ints_bytes: bytes = b"".join(encode_int(8, int(coerce_int(x))) for x in val)
        return attr_bytes + ints_bytes + encode_int(20, 7)  # INTS
    if val and isinstance(val[0], float):
        floats_bytes: bytes = b"".join(encode_tag(7, 5) + struct.pack("<f", float(coerce_float(x))) for x in val)
        return attr_bytes + floats_bytes + encode_int(20, 6)  # FLOATS
    if val and isinstance(val[0], str):
        strs_bytes: bytes = b"".join(encode_length_delimited(9, str(x).encode("utf-8")) for x in val)
        return attr_bytes + strs_bytes + encode_int(20, 8)  # STRINGS
    return attr_bytes + encode_int(20, 7)


def serialize_attribute(
    name: str, val: BackendArg, graph_serializer: Callable[[IRGraph, str], bytes] | None = None
) -> bytes:
    """Serialize a single node attribute into AttributeProto bytes.

    Args:
        name (str): Attribute key name.
        val (BackendArg): Attribute payload value.
        graph_serializer (Callable[[IRGraph, str], bytes] | None): Subgraph serializer callback.

    Returns:
        bytes: Wire bytes for AttributeProto message.
    """
    attr_bytes: bytes = encode_string(1, name)
    if isinstance(val, float):
        return attr_bytes + encode_float(2, val) + encode_int(20, 1)  # FLOAT
    if isinstance(val, int) and not isinstance(val, bool):
        return attr_bytes + encode_int(3, val) + encode_int(20, 2)  # INT
    if isinstance(val, (str, bytes, bytearray)):
        raw_str = val.encode("utf-8") if isinstance(val, str) else bytes(val)
        return attr_bytes + encode_length_delimited(4, raw_str) + encode_int(20, 3)  # STRING
    if isinstance(val, (list, tuple)):
        return _serialize_sequence_attribute(attr_bytes, val)
    if adapt_backend_module(val).has_symbol("nodes") and graph_serializer is not None:
        subgraph_bytes: bytes = graph_serializer(ensure_graph(val), f"{name}_subgraph")
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
