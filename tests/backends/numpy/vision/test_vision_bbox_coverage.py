"""Tests for test_vision_bbox_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.numpy.eager import vision_bbox as vision_bbox


def test_vision_bbox_module() -> None:
    """Verify vision bbox module structure and exports."""
    assert hasattr(vision_bbox, "__all__")
    assert vision_bbox.__all__ == []
