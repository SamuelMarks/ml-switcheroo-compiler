"""Tests for test_vision_bbox_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.numpy.eager import vision_bbox as vision_bbox
from ml_switcheroo_compiler.core.shape import _has_attr


def test_vision_bbox_module() -> None:
    """Verify vision bbox module structure and exports."""
    assert _has_attr(vision_bbox, "__all__")
    assert vision_bbox.__all__ == []
