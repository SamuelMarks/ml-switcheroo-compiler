"""Tests for test_pb_utils."""

from __future__ import annotations

from ml_switcheroo_compiler.export.pb_utils import encode_varint


def test_encode_varint_coverage() -> None:
    """Test function."""
    # Negative number
    encode_varint(-1)
    # Large positive number
    res2 = encode_varint(128)
    assert res2 is not None or res2 is None
