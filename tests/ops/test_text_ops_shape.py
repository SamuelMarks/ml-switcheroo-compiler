"""Test text operations shape inference."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.text import CreateToken


def test_create_token_shape() -> None:
    """Test CreateToken shape inference."""
    op = CreateToken()

    a = type("MockTensor", (), {"shape": (4, 5, 6)})()

    # Default (no seq_length)
    assert op.infer_shape(a) == (4, 5, 6)

    # max_length
    assert op.infer_shape(a, max_length=128) == (4, 5, 6, 128)
