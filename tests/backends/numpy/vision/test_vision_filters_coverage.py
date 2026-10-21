"""Tests for test_vision_filters_coverage."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_filters as vision_filters
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    pass


def test_vision_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test vision filter operations directly and through eager registry.

    Args:
        monkeypatch: Test parameter `monkeypatch`.
    """
    images = np.ones((1, 8, 8, 3), dtype=np.float32)

    monkeypatch.setattr(vision_filters, "_np_gaussian_blur", lambda mod, img, **kw: img)

    res_blur = vision_filters._np_random_gaussian_blur(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert as_tensor(getattr(res_blur, "shape")) == as_tensor(getattr(images, "shape"))

    reg_blur = numpy_eager_registry.get("RandomGaussianBlur")
    res_reg_blur = as_callable(reg_blur)(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert as_tensor(getattr(res_reg_blur, "shape")) == as_tensor(getattr(images, "shape"))

    res_sharp = vision_filters._np_random_sharpness(np, images, factor=1.5)
    assert as_tensor(getattr(res_sharp, "shape")) == as_tensor(getattr(images, "shape"))

    reg_sharp = numpy_eager_registry.get("RandomSharpness")
    res_reg_sharp = as_callable(reg_sharp)(np, images, factor=2.0)
    assert as_tensor(getattr(res_reg_sharp, "shape")) == as_tensor(getattr(images, "shape"))
