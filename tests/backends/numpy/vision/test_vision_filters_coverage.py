"""Tests for test_vision_filters_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_filters as vision_filters

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_vision_filters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test vision filter operations directly and through eager registry."""
    images = np.ones((1, 8, 8, 3), dtype=np.float32)

    monkeypatch.setattr(vision_filters, "_np_gaussian_blur", lambda mod, img, **kw: img)

    res_blur = vision_filters._np_random_gaussian_blur(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert cast("EagerTensorProtocol", getattr(res_blur, "shape")) == cast("EagerTensorProtocol", getattr(images, "shape"))

    reg_blur = numpy_eager_registry.get("RandomGaussianBlur")
    res_reg_blur = cast("Callable[..., EagerTensorProtocol]", reg_blur)(np, images, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert cast("EagerTensorProtocol", getattr(res_reg_blur, "shape")) == cast("EagerTensorProtocol", getattr(images, "shape"))

    res_sharp = vision_filters._np_random_sharpness(np, images, factor=1.5)
    assert cast("EagerTensorProtocol", getattr(res_sharp, "shape")) == cast("EagerTensorProtocol", getattr(images, "shape"))

    reg_sharp = numpy_eager_registry.get("RandomSharpness")
    res_reg_sharp = cast("Callable[..., EagerTensorProtocol]", reg_sharp)(np, images, factor=2.0)
    assert cast("EagerTensorProtocol", getattr(res_reg_sharp, "shape")) == cast("EagerTensorProtocol", getattr(images, "shape"))
