"""Tests for test_vision_frontend_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.vision.frontend as vision_frontend
from ml_switcheroo_compiler.core.shape import _has_attr


def test_vision_frontend_module() -> None:
    """Verify vision frontend module structure and exports."""
    assert _has_attr(vision_frontend, "__all__")
    assert vision_frontend.__all__ == []
