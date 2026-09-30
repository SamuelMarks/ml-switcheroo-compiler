"""Tests for test_vision_filters_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_filters as vision_filters


def test_vision_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test vision filter operations directly and through eager registry."""
    images = np.ones((1, 8, 8, 3), dtype=np.float32)

    monkeypatch.setattr(vision_filters, "_np_gaussian_blur", lambda mod, img, **kw: img)

    res_blur = vision_filters._np_random_gaussian_blur(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert res_blur.shape == images.shape

    reg_blur = numpy_eager_registry.get("RandomGaussianBlur")
    res_reg_blur = reg_blur(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert res_reg_blur.shape == images.shape

    res_sharp = vision_filters._np_random_sharpness(np, images, factor=1.5)
    assert res_sharp.shape == images.shape

    reg_sharp = numpy_eager_registry.get("RandomSharpness")
    res_reg_sharp = reg_sharp(np, images, factor=2.0)
    assert res_reg_sharp.shape == images.shape
