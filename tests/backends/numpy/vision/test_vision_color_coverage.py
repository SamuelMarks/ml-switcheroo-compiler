"""Tests for test_vision_color_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import vision_color as color_mod


def test_vision_color_ops() -> None:
    """Test all vision color transformations in vision_color.

    Returns:
        None
    """
    img = np.array([[[[0.2, 0.4, 0.6], [0.8, 0.5, 0.1]]]], dtype=np.float32)

    # AdjustBrightness
    b_adj = color_mod._np_adjust_brightness(np, img, 0.1)
    assert np.allclose(b_adj[0, 0, 0], [0.3, 0.5, 0.7])
    assert numpy_eager_registry.get("AdjustBrightness")(np, img, 0.1) is not None

    # AdjustContrast
    c_adj = color_mod._np_adjust_contrast(np, img, 1.5)
    assert c_adj.shape == img.shape
    assert numpy_eager_registry.get("AdjustContrast")(np, img, 1.5) is not None

    # AdjustHue
    h_adj = color_mod._np_adjust_hue(np, img, 0.2)
    assert np.array_equal(h_adj, img)
    assert numpy_eager_registry.get("AdjustHue")(np, img, 0.2) is not None

    # AdjustSaturation
    s_adj = color_mod._np_adjust_saturation(np, img, 1.2)
    assert s_adj.shape == img.shape
    assert numpy_eager_registry.get("AdjustSaturation")(np, img, 1.2) is not None

    # AutoContrast: non-constant diff vs constant diff
    ac1 = color_mod._np_auto_contrast(np, img, value_range=(0, 255))
    assert ac1.shape == img.shape

    const_img = np.ones((1, 2, 2, 3), dtype=np.float32) * 0.5
    ac_const = color_mod._np_auto_contrast(np, const_img)
    assert ac_const.shape == const_img.shape
    assert numpy_eager_registry.get("AutoContrast")(np, img) is not None

    # Equalization: cdf_m diff == 0 vs diff != 0
    zero_img = np.zeros((1, 2, 2, 1), dtype=np.float32)
    eq_zero = color_mod._np_equalization(np, zero_img)
    assert np.array_equal(eq_zero, zero_img)

    grad_img = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape((1, 4, 4, 1))
    eq_grad = color_mod._np_equalization(np, grad_img)
    assert eq_grad.shape == grad_img.shape
    assert numpy_eager_registry.get("Equalization")(np, grad_img) is not None

    # Invert
    inv = color_mod._np_invert(np, img, value_range=(0, 255))
    assert inv.shape == img.shape
    assert numpy_eager_registry.get("Invert")(np, img) is not None

    # Posterize
    post = color_mod._np_posterize(np, img, bits=4)
    assert post.shape == img.shape
    assert numpy_eager_registry.get("Posterize")(np, img) is not None

    # RgbToGrayscale
    gray = color_mod._np_rgb_to_grayscale(np, img, data_format="channels_last")
    assert gray.shape == (1, 1, 2, 1)
    assert numpy_eager_registry.get("RgbToGrayscale")(np, img) is not None

    # Solarize
    sol = color_mod._np_solarize(np, img, threshold=0.5, value_range=(0, 1))
    assert sol.shape == img.shape
    assert numpy_eager_registry.get("Solarize")(np, img) is not None
