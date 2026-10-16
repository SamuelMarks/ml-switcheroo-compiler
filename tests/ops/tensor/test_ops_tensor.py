"""Tests for test_ops_tensor."""

from __future__ import annotations

# ruff: noqa: E501
from ml_switcheroo_compiler.ops.tensor import allclose, take_along_axis


def test_tensor_exports() -> None:
    """Test function test_tensor_exports."""
    assert allclose is not None
    assert take_along_axis is not None
