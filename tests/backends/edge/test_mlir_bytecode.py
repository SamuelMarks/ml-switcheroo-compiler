"""Tests for mlir_bytecode."""

from ml_switcheroo_compiler.backends.edge.mlir_bytecode import MLIRBytecodeEncoder


def test_mlir_bytecode_encoder():
    """Test MLIRBytecodeEncoder basic operations."""
    encoder = MLIRBytecodeEncoder()
    assert encoder.strings == ["builtin", "func", "stablehlo"]

    # Test adding strings
    idx1 = encoder._add_string("hello")
    idx2 = encoder._add_string("world")
    idx3 = encoder._add_string("hello")
    assert idx1 == 3
    assert idx2 == 4
    assert idx3 == 3
    assert encoder.strings == ["builtin", "func", "stablehlo", "hello", "world"]

    # Test add dialect
    encoder.add_dialect("func")
    encoder.add_dialect("tensor")
    encoder.add_dialect("func")  # duplicate

    # Test add op
    encoder.add_op("func.return", ["%0"], [])

    # Test varint
    assert encoder._encode_varint(0) == b"\x00"
    assert encoder._encode_varint(127) == b"\x7f"
    assert encoder._encode_varint(128) == b"\x80\x01"
    assert encoder._encode_varint(300) == b"\xac\x02"

    # Test encoding sections and final output
    bytecode = encoder.encode()
    assert bytecode.startswith(b"ML\xefR\x01")


def test_mlir_bytecode_encoder_ssa_and_types():
    """Test MLIRBytecodeEncoder SSA value tracking and type section encoding."""
    encoder = MLIRBytecodeEncoder()

    # Register custom types
    t0 = encoder.add_type("tensor<4x4xf32>")
    t1 = encoder.add_type("tensor<4x4xi32>")
    assert t0 == 0
    assert t1 == 1
    assert encoder.add_type("tensor<4x4xf32>") == 0  # Deduplication

    # Add ops with chained inputs/outputs and explicit types
    encoder.add_op("stablehlo.constant", args=[], rets=["c0"], ret_types=["tensor<4x4xf32>"])
    encoder.add_op("stablehlo.add", args=["c0", "c0"], rets=["out0"], ret_types=["tensor<4x4xf32>"])

    # Check SSA mappings are assigned monotonically
    assert encoder.ssa_values["c0"] == 1
    assert encoder.ssa_values["out0"] == 2

    op_add = encoder.ops[1]
    assert op_add["operand_ssa_ids"] == [1, 1]
    assert op_add["result_ssa_ids"] == [2]
    assert op_add["result_type_indices"] == [0]

    # Verify bytecode encodes all 4 sections (STRING: 0, DIALECT: 1, ATTR_TYPE: 2, IR: 3)
    bytecode = encoder.encode()
    assert len(bytecode) > 20
    assert bytecode.startswith(b"ML\xefR\x01")

    # Check section IDs exist in bytecode payload
    assert b"\x00" in bytecode  # STRING section header
    assert b"\x01" in bytecode  # DIALECT section header
    assert b"\x02" in bytecode  # ATTR_TYPE section header
    assert b"\x03" in bytecode  # IR section header
