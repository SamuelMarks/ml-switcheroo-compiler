"""Tests for test_vision_frontend_coverage."""

from __future__ import annotations

import ml_switcheroo_compiler.ops.vision.frontend as vision_frontend
from tests.testing_utils import ensure_eager_tensor


def test_vision_frontend_module() -> None:
    """Verify vision frontend module structure and exports."""
    assert ensure_eager_tensor(vision_frontend).__all__ is not None
    assert vision_frontend.__all__ == []
