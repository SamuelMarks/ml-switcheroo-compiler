"""Unit tests for Protobuf varint encoding and binary wire writer utilities."""

from __future__ import annotations

from ml_switcheroo_compiler.export.pb_utils import ProtobufWriter, encode_varint


def test_encode_varint_values() -> None:
    """Verify `encode_varint` encodes zero, single-byte, multi-byte, and negative 64-bit integers."""
    assert encode_varint(0) == b"\x00"
    assert encode_varint(1) == b"\x01"
    assert encode_varint(128) == b"\x80\x01"
    assert encode_varint(300) == b"\xac\x02"
    neg_bytes = encode_varint(-1)
    assert neg_bytes == b"\xff\xff\xff\xff\xff\xff\xff\xff\xff\x01"
    assert len(neg_bytes) == 10


def test_protobuf_writer_wire_encoding() -> None:
    """Verify `ProtobufWriter` serializes varint, bytes, string, and nested message fields accurately."""
    writer = ProtobufWriter()
    writer.add_varint(1, 42)
    writer.add_bytes(2, b"raw_bytes")
    writer.add_string(3, "test_string")

    nested = ProtobufWriter()
    nested.add_varint(1, 100)
    writer.add_message(4, nested)

    expected = b"\x08\x2a\x12\x09raw_bytes\x1a\x0btest_string\x22\x02\x08\x64"
    assert writer.get_bytes() == expected
