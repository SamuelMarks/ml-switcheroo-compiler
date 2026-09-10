"""Exhaustive tests for pure-Python zero-dependency ONNX binary protobuf serialization."""

import os
import struct
import tempfile

import onnx

from ml_switcheroo_compiler.backends.edge.onnx import (
    ONNXCodeGenerator,
    encode_float,
    encode_int,
    encode_length_delimited,
    encode_string,
    encode_tag,
    encode_varint,
    serialize_attribute,
    serialize_dimension,
    serialize_tensor_shape,
    serialize_type_proto,
    serialize_value_info,
)
from ml_switcheroo_compiler.ir.core import IRGraph, LogicalNode


def test_protobuf_wire_primitives():
    """Test low-level Protocol Buffers wire encoding primitives."""
    # Test varint
    assert encode_varint(0) == b"\x00"
    assert encode_varint(1) == b"\x01"
    assert encode_varint(127) == b"\x7f"
    assert encode_varint(128) == b"\x80\x01"
    assert encode_varint(300) == b"\xac\x02"

    # Test tag
    assert encode_tag(1, 0) == b"\x08"  # (1 << 3) | 0 = 8
    assert encode_tag(1, 2) == b"\x0a"  # (1 << 3) | 2 = 10
    assert encode_tag(2, 5) == b"\x15"  # (2 << 3) | 5 = 21

    # Test string and length delimited
    assert encode_string(1, "abc") == b"\x0a\x03abc"
    assert encode_length_delimited(2, b"\x01\x02") == b"\x12\x02\x01\x02"

    # Test int (positive and negative)
    assert encode_int(1, 42) == b"\x08\x2a"
    neg_bytes = encode_int(1, -1)
    assert len(neg_bytes) == 11  # tag + 10 bytes varint for 64-bit two's complement

    # Test float
    f_bytes = encode_float(2, 1.5)
    assert f_bytes[:1] == b"\x15"
    assert struct.unpack("<f", f_bytes[1:])[0] == 1.5


def test_protobuf_spec_structures():
    """Test serialization of dimensions, shapes, type protos, and value infos."""
    # Fixed dimension
    dim_fixed = serialize_dimension(16)
    assert dim_fixed == encode_int(1, 16)

    # Symbolic / dynamic dimension
    dim_sym = serialize_dimension("batch")
    assert dim_sym == encode_string(2, "batch")
    dim_none = serialize_dimension(None)
    assert dim_none == encode_string(2, "batch_size")

    # Tensor shape
    shape_bytes = serialize_tensor_shape([16, "seq_len"])
    assert len(shape_bytes) > 0

    # Type proto & value info
    type_bytes = serialize_type_proto(1, [16, "seq_len"])
    assert len(type_bytes) > 0

    vi_bytes = serialize_value_info("X", 1, [16, "seq_len"])
    assert len(vi_bytes) > 0

    # Attribute with various types
    attr_f = serialize_attribute("alpha", 0.5)
    assert len(attr_f) > 0
    attr_i = serialize_attribute("axis", 1)
    assert len(attr_i) > 0
    attr_s = serialize_attribute("mode", "constant")
    assert len(attr_s) > 0
    attr_bytes = serialize_attribute("raw", b"data")
    assert len(attr_bytes) > 0
    attr_ints = serialize_attribute("pads", [1, 2, 3, 4])
    assert len(attr_ints) > 0
    attr_floats = serialize_attribute("weights", [0.1, 0.2])
    assert len(attr_floats) > 0
    attr_strs = serialize_attribute("labels", ["a", "b"])
    assert len(attr_strs) > 0
    attr_empty = serialize_attribute("empty", [])
    assert len(attr_empty) > 0


def test_pure_python_onnx_binary_export_and_checker():
    """Test generating a complete ONNX model with pure-Python serialization and validating via onnx.checker."""
    g = IRGraph(name="test_binary_model")

    in_x = LogicalNode(id="X", op_type="Input", shape_metadata=(2, 4), attributes={"dtype": "float32"})
    in_w = LogicalNode(id="W", op_type="Input", shape_metadata=(4, 8), attributes={"dtype": "float32"})
    cst_b = LogicalNode(
        id="B",
        op_type="Constant",
        shape_metadata=(1, 8),
        attributes={"dtype": "float32", "value": 0.5},
    )
    matmul_node = LogicalNode(id="matmul_out", op_type="MatMul", inputs=["X", "W"], shape_metadata=(2, 8))
    add_node = LogicalNode(id="Y", op_type="Add", inputs=["matmul_out", "B"], shape_metadata=(2, 8))

    g.nodes = {"X": in_x, "W": in_w, "B": cst_b, "matmul_out": matmul_node, "Y": add_node}
    g.inputs = ["X", "W"]
    g.outputs = ["Y"]
    g.sorted_nodes = [in_x, in_w, cst_b, matmul_node, add_node]

    gen = ONNXCodeGenerator(g)
    binary_data = gen.serialize_model_to_bytes()

    assert isinstance(binary_data, bytes)
    assert len(binary_data) > 0

    # Validate against standard ONNX runtime / checker
    model_proto = onnx.load_model_from_string(binary_data)
    onnx.checker.check_model(model_proto)

    assert model_proto.producer_name == "ml-switcheroo-compiler"
    assert model_proto.opset_import[0].version == 18
    assert len(model_proto.graph.node) == 3
    assert len(model_proto.graph.input) == 2
    assert len(model_proto.graph.output) == 1

    # Export to disk
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tf:
        tf.close()
        gen.export_onnx(tf.name)
        with open(tf.name, "rb") as f:
            disk_bytes = f.read()
        assert disk_bytes == binary_data
        os.unlink(tf.name)


def test_onnx_binary_dynamic_axes():
    """Test dynamic axes propagation in binary ONNX models."""
    g = IRGraph(name="test_dyn_axes")

    in_x = LogicalNode(id="X", op_type="Input", shape_metadata=(1, 10), attributes={"dtype": "float32"})
    relu_node = LogicalNode(id="Y", op_type="Relu", inputs=["X"], shape_metadata=(1, 10))

    g.nodes = {"X": in_x, "Y": relu_node}
    g.inputs = ["X"]
    g.outputs = ["Y"]
    g.sorted_nodes = [in_x, relu_node]

    gen = ONNXCodeGenerator(g)
    dynamic_axes = {"X": {0: "batch_size"}, "Y": {0: "batch_size"}}
    binary_data = gen.serialize_model_to_bytes(dynamic_axes=dynamic_axes)

    model_proto = onnx.load_model_from_string(binary_data)
    onnx.checker.check_model(model_proto)

    # Verify dynamic dimension string was set
    dim_in = model_proto.graph.input[0].type.tensor_type.shape.dim[0]
    assert dim_in.dim_param == "batch_size"

    dim_out = model_proto.graph.output[0].type.tensor_type.shape.dim[0]
    assert dim_out.dim_param == "batch_size"


def test_onnx_binary_control_flow_subgraphs():
    """Test binary serialization of If control flow subgraphs."""
    main_g = IRGraph(name="main_g")

    cond_node = LogicalNode(id="cond", op_type="Input", shape_metadata=(), attributes={"dtype": "bool"})
    in_x = LogicalNode(id="X", op_type="Input", shape_metadata=(2, 2), attributes={"dtype": "float32"})

    # Then subgraph: Add(X, X) -> then_out
    then_g = IRGraph(name="then_g")
    t_in = LogicalNode(id="X", op_type="Input", shape_metadata=(2, 2))
    t_add = LogicalNode(id="then_out", op_type="Add", inputs=["X", "X"], shape_metadata=(2, 2))
    then_g.nodes = {"X": t_in, "then_out": t_add}
    then_g.inputs = ["X"]
    then_g.outputs = ["then_out"]
    then_g.sorted_nodes = [t_in, t_add]

    # Else subgraph: Sub(X, X) -> else_out
    else_g = IRGraph(name="else_g")
    e_in = LogicalNode(id="X", op_type="Input", shape_metadata=(2, 2))
    e_sub = LogicalNode(id="else_out", op_type="Sub", inputs=["X", "X"], shape_metadata=(2, 2))
    else_g.nodes = {"X": e_in, "else_out": e_sub}
    else_g.inputs = ["X"]
    else_g.outputs = ["else_out"]
    else_g.sorted_nodes = [e_in, e_sub]

    if_node = LogicalNode(
        id="Y",
        op_type="If",
        inputs=["cond"],
        shape_metadata=(2, 2),
        attributes={"then_branch": then_g, "else_branch": else_g},
    )

    main_g.nodes = {"cond": cond_node, "X": in_x, "Y": if_node}
    main_g.inputs = ["cond", "X"]
    main_g.outputs = ["Y"]
    main_g.sorted_nodes = [cond_node, in_x, if_node]

    gen = ONNXCodeGenerator(main_g)
    binary_data = gen.serialize_model_to_bytes()

    model_proto = onnx.load_model_from_string(binary_data)
    onnx.checker.check_model(model_proto)

    assert len(model_proto.graph.node) == 1
    assert model_proto.graph.node[0].op_type == "If"
    attr_names = [a.name for a in model_proto.graph.node[0].attribute]
    assert "then_branch" in attr_names
    assert "else_branch" in attr_names


def test_onnx_binary_loop_subgraph():
    """Test binary serialization of Loop control flow subgraphs."""
    main_g = IRGraph(name="main_loop_g")
    trip_count = LogicalNode(id="trip_count", op_type="Input", shape_metadata=(), attributes={"dtype": "int64"})
    cond_node = LogicalNode(id="cond", op_type="Input", shape_metadata=(), attributes={"dtype": "bool"})
    init_v = LogicalNode(id="v", op_type="Input", shape_metadata=(1,), attributes={"dtype": "float32"})

    body_g = IRGraph(name="body_g")
    b_iter = LogicalNode(id="iter", op_type="Input", shape_metadata=(), attributes={"dtype": "int64"})
    b_cond = LogicalNode(id="cond_in", op_type="Input", shape_metadata=(), attributes={"dtype": "bool"})
    b_v = LogicalNode(id="v_in", op_type="Input", shape_metadata=(1,), attributes={"dtype": "float32"})
    b_add = LogicalNode(id="v_out", op_type="Add", inputs=["v_in", "v_in"], shape_metadata=(1,))
    body_g.nodes = {"iter": b_iter, "cond_in": b_cond, "v_in": b_v, "v_out": b_add}
    body_g.inputs = ["iter", "cond_in", "v_in"]
    body_g.outputs = ["cond_in", "v_out"]
    body_g.sorted_nodes = [b_iter, b_cond, b_v, b_add]

    loop_node = LogicalNode(
        id="loop_res",
        op_type="Loop",
        inputs=["trip_count", "cond", "v"],
        shape_metadata=(1,),
        attributes={"body": body_g},
    )

    main_g.nodes = {"trip_count": trip_count, "cond": cond_node, "v": init_v, "loop_res": loop_node}
    main_g.inputs = ["trip_count", "cond", "v"]
    main_g.outputs = ["loop_res"]
    main_g.sorted_nodes = [trip_count, cond_node, init_v, loop_node]

    gen = ONNXCodeGenerator(main_g)
    binary_data = gen.serialize_model_to_bytes()

    assert len(binary_data) > 0


def test_onnx_additional_edges_and_validation() -> None:
    """Test topological sort Input node handling, decode_varint edge cases, and protobuf wire types."""
    import pytest

    from ml_switcheroo_compiler.backends.edge.onnx import (
        decode_varint,
        validate_onnx_model_bytes,
        validate_topological_sort,
    )

    # 1. validate_topological_sort with Input node and invalid ordering (line 340)
    in_node = LogicalNode(id="x", op_type="Input")
    add_node = LogicalNode(id="y", op_type="Add", inputs=["x"])
    assert validate_topological_sort([in_node, add_node], graph_inputs=[]) is True

    bad_order_node = LogicalNode(id="z", op_type="Add", inputs=["unseen_input"])
    with pytest.raises(ValueError, match="Graph nodes are not topologically sorted"):
        validate_topological_sort([bad_order_node], graph_inputs=[])

    # 2. decode_varint multi-byte and loop termination without break (lines 357->364, 363)
    val, off = decode_varint(encode_varint(1000), 0)
    assert val == 1000
    # Truncated varint with continuation bit set on final byte (reaches while loop termination without break)
    val_trunc, off_trunc = decode_varint(b"\x80", 0)
    assert off_trunc == 1

    # 3. validate_onnx_model_bytes empty or too short (line 380)
    with pytest.raises(ValueError, match="empty or too short"):
        validate_onnx_model_bytes(b"")
    with pytest.raises(ValueError, match="empty or too short"):
        validate_onnx_model_bytes(b"\x08\x01")

    # 4. validate_onnx_model_bytes wire types and subfields in field 8 (lines 412-419, 422, 426-431)
    # Construct a binary payload containing:
    # - field 1 (ir_version = 8)
    # - field 2 (producer_name = "test")
    # - field 7 (graph = b"dummy_graph")
    # - field 8 (opset import): subfield 2 (opset_version = 17), subfield with wt 0, subfield with wt 2, subfield with other wt (break)
    # - field 99 with wire type 0 (varint)
    # - field 100 with wire type 2 (length-delimited)
    # - field 101 with wire type 5 (32-bit fixed)
    # - field 102 with wire type 1 (64-bit fixed)
    f1 = encode_tag(1, 0) + encode_varint(8)
    f2 = encode_string(2, "test_prod")
    f7 = encode_length_delimited(7, b"valid_graph_payload")

    # field 8 subfields
    sub_f2 = encode_tag(2, 0) + encode_varint(17)
    sub_wt0 = encode_tag(3, 0) + encode_varint(99)
    sub_wt2 = encode_length_delimited(4, b"meta")
    sub_other_wt = encode_tag(5, 5) + b"\x00\x00\x00\x00"  # wire type 5 inside opset import triggers else break
    f8 = encode_length_delimited(8, sub_f2 + sub_wt0 + sub_wt2 + sub_other_wt)

    # Top-level wire types
    f_wt0 = encode_tag(99, 0) + encode_varint(123)
    f_wt2 = encode_length_delimited(100, b"extra")
    f_wt5 = encode_tag(101, 5) + b"\x01\x02\x03\x04"
    f_wt1 = encode_tag(102, 1) + b"\x01\x02\x03\x04\x05\x06\x07\x08"

    payload = f1 + f2 + f7 + f8 + f_wt0 + f_wt2 + f_wt5 + f_wt1
    meta = validate_onnx_model_bytes(payload)
    assert meta["ir_version"] == 8
    assert meta["producer_name"] == "test_prod"
    assert meta["opset_version"] == 17

    # Malformed top-level wire type (line 431)
    bad_payload = f1 + f7 + encode_tag(103, 7)
    with pytest.raises(ValueError, match="Malformed protobuf wire type"):
        validate_onnx_model_bytes(bad_payload)

    # Missing graph payload (line 434)
    missing_graph_payload = f1 + f2 + f8
    with pytest.raises(ValueError, match="missing a valid GraphProto payload"):
        validate_onnx_model_bytes(missing_graph_payload)

    # 5. ONNXCodeGenerator opset_version bounds (line 703)
    simple_g = IRGraph(name="bounds_test")
    in_a = LogicalNode(id="a", op_type="Input", shape_metadata=(1,))
    out_b = LogicalNode(id="b", op_type="Identity", inputs=["a"], shape_metadata=(1,))
    simple_g.nodes = {"a": in_a, "b": out_b}
    simple_g.inputs = ["a"]
    simple_g.outputs = ["b"]
    simple_g.sorted_nodes = [in_a, out_b]

    gen = ONNXCodeGenerator(simple_g)
    with pytest.raises(ValueError, match="Unsupported ONNX opset_version 10"):
        gen.serialize_model_to_bytes(opset_version=10)
    with pytest.raises(ValueError, match="Unsupported ONNX opset_version 25"):
        gen.serialize_model_to_bytes(opset_version=25)

    # 6. ONNXCodeGenerator.verify (lines 891-897)
    assert gen.verify() is True

    # verify with file path
    with tempfile.NamedTemporaryFile("wb", suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        gen.export_onnx(tmp_path)
        assert gen.verify(tmp_path) is True
        # verify with non-existent file path falls back to in-memory serialization
        assert gen.verify("/non/existent/path/model.onnx") is True
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
