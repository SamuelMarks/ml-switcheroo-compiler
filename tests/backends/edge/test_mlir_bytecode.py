"""Tests for mlir_bytecode."""

from ml_switcheroo_compiler.backends.edge.mlir_bytecode import MLIRBytecodeEncoder


def test_mlir_bytecode_encoder():
    """Test MLIRBytecodeEncoder."""
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
