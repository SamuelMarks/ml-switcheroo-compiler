"""Tests for test_vision_filtering_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_filtering as vision_filtering_module
from ml_switcheroo_compiler.ops.configs import BlurConfig


def test_vision_filtering_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test vision filtering operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture for monkeypatching.

    Returns:
        None
    """
    for item in vision_filtering_module.__all__:
        assert hasattr(vision_filtering_module, item)

    imgs = np.ones((1, 8, 8, 3), dtype=np.float32)
    assert np.array_equal(vision_filtering_module._np_degeneration(np, imgs), imgs)
    reg_degen = numpy_eager_registry.get("Degeneration")
    assert np.array_equal(reg_degen(np, imgs), imgs)

    assert np.array_equal(vision_filtering_module._np_sharpen(np, imgs), imgs)
    reg_sharp = numpy_eager_registry.get("Sharpen")
    assert np.array_equal(reg_sharp(np, imgs), imgs)

    monkeypatch.setattr(vision_filtering_module, "gaussian_blur_eager", lambda mod, img, cfg: img * 0.5)
    monkeypatch.setattr(vision_filtering_module, "median_filter_eager", lambda mod, img, **kw: img * 0.25)

    # GaussianBlur with dict config
    blur_dict = vision_filtering_module._np_gaussian_blur(np, imgs, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert np.allclose(blur_dict, imgs * 0.5)

    # GaussianBlur with BlurConfig
    cfg_obj = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    blur_obj = vision_filtering_module._np_gaussian_blur(np, imgs, config=cfg_obj)
    assert np.allclose(blur_obj, imgs * 0.5)

    reg_blur = numpy_eager_registry.get("GaussianBlur")
    assert np.allclose(reg_blur(np, imgs, config=cfg_obj), imgs * 0.5)

    # MedianFilter
    med_res = vision_filtering_module._np_median_filter(np, imgs)
    assert np.allclose(med_res, imgs * 0.25)

    reg_med = numpy_eager_registry.get("MedianFilter")
    assert np.allclose(reg_med(np, imgs), imgs * 0.25)
