"""Tests for test_vision_bbox_coverage."""

from __future__ import annotations

from ml_switcheroo_compiler.backends.numpy.eager import vision_bbox as vision_bbox
from tests.testing_utils import ensure_eager_tensor


def test_vision_bbox_module() -> None:
    """Verify vision bbox module structure and exports."""
    assert ensure_eager_tensor(vision_bbox).__all__ is not None
    assert vision_bbox.__all__ == []
