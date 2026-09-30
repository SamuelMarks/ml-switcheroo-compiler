"""Tests for test_pb_utils_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.export.pb_utils as pb_utils_mod


class _DummyBadShape:
    """Shape object whose iterator raises TypeError."""

    def __iter__(self) -> _DummyBadShape:
        """Return self as iterator.

        Returns:
            _DummyBadShape: Self.
        """
        return self

    def __next__(self) -> int:
        """Raise TypeError on iteration.

        Raises:
            TypeError: Simulated invalid dimension.
        """
        raise TypeError("Not iterable dimension")


def test_pb_utils_exhaustive() -> None:
    """Verify varint encoding and protobuf writer operations."""
    # 1. encode_varint with zero, positive, and negative numbers
    assert pb_utils_mod.encode_varint(0) == b"\x00"
    assert pb_utils_mod.encode_varint(1) == b"\x01"
    assert pb_utils_mod.encode_varint(300) == b"\xac\x02"
    neg_bytes = pb_utils_mod.encode_varint(-1)
    assert len(neg_bytes) == 10

    # 2. ProtobufWriter fields
    writer = pb_utils_mod.ProtobufWriter()
    writer.add_varint(1, 42)
    writer.add_bytes(2, b"raw_bytes")
    writer.add_string(3, "test_string")

    nested = pb_utils_mod.ProtobufWriter()
    nested.add_varint(1, 100)
    writer.add_message(4, nested)

    data = writer.get_bytes()
    assert len(data) > 0
    assert b"raw_bytes" in data
    assert b"test_string" in data
