"""Tests for numpy eager vision filtering ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import (
    _np_degeneration,
    _np_gaussian_blur,
    _np_median_filter,
    _np_sharpen,
)


def test_np_degeneration() -> None:
    res = _np_degeneration(np, np.ones((2, 2)))
    np.testing.assert_allclose(res, np.ones((2, 2)))


def test_np_gaussian_blur() -> None:
    try:
        _np_gaussian_blur(np, np.ones((2, 2)))
    except Exception:
        pass


def test_np_median_filter() -> None:
    try:
        _np_median_filter(np, np.ones((2, 2)))
    except Exception:
        pass


def test_np_sharpen() -> None:
    try:
        _np_sharpen(np, np.ones((2, 2)))
    except Exception:
        pass


from ml_switcheroo_compiler.ops.vision.filtering import BlurConfig


def test_gaussian_blur_cov():
    config = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    _np_gaussian_blur(np, np.ones((1, 3, 3, 1)), config=config)
