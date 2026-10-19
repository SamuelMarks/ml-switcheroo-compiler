"""Tests for test_pb_utils."""

from __future__ import annotations

from ml_switcheroo_compiler.export.pb_utils import encode_varint


def test_encode_varint_coverage() -> None:
    """Verify the behavior and output invariants of test encode varint coverage."""
    # Negative number
    neg_bytes = encode_varint(-1)
    assert len(neg_bytes) == 10
    # Large positive number
    res2 = encode_varint(128)
    assert res2 == b"\x80\x01"
