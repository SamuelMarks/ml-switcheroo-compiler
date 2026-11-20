"""Unit tests for NumPy eager random Gaussian blur and sharpness vision filters."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.vision_ops.filtering import (
    _np_gaussian_blur,
    _np_random_gaussian_blur,
    _np_random_sharpness,
)
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry


def test_random_gaussian_blur_real_execution() -> None:
    """Verify `_np_random_gaussian_blur` and `RandomGaussianBlur` execute real 2D Gaussian filtering."""
    mod = np
    impulse = np.zeros((1, 2, 5, 5), dtype=np.float32)
    impulse[:, :, 2, 2] = 1.0

    res_blur = _np_random_gaussian_blur(mod, impulse, kernel_size=(3, 3), sigma=(1.0, 1.0))
    expected = _np_gaussian_blur(mod, impulse, kernel_size=(3, 3), sigma=(1.0, 1.0))
    assert res_blur.shape == (1, 2, 5, 5)
    np.testing.assert_allclose(res_blur, expected, atol=1e-6)
    assert np.isclose(float(np.sum(res_blur[0, 0])), 1.0, atol=1e-5)
    assert float(res_blur[0, 0, 2, 2]) > float(res_blur[0, 0, 2, 1]) > 0.0

    reg_blur = numpy_eager_registry.get("RandomGaussianBlur")
    assert reg_blur is not None
    res_reg = np.asarray(reg_blur(np, impulse, kernel_size=(3, 3), sigma=(1.0, 1.0)))
    np.testing.assert_allclose(res_reg, expected, atol=1e-6)


def test_random_sharpness_execution() -> None:
    """Verify `_np_random_sharpness` and `RandomSharpness` preserve spatial layout and values."""
    mod = np
    images = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)

    res_sharp = _np_random_sharpness(mod, images, factor=1.5)
    assert res_sharp.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(res_sharp, images)

    reg_sharp = numpy_eager_registry.get("RandomSharpness")
    assert reg_sharp is not None
    res_reg_sharp = np.asarray(reg_sharp(np, images, factor=2.0))
    assert res_reg_sharp.shape == (1, 1, 4, 4)
    np.testing.assert_allclose(res_reg_sharp, images)
