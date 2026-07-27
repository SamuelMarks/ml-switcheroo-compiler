"""Tests for numpy eager vision filters."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_filters import (
    _np_random_gaussian_blur,
    _np_random_sharpness,
)


def test_random_gaussian_blur() -> None:
    """Test random gaussian blur.

    Returns:
        None
    """
    images = np.ones((1, 5, 5, 3))
    kernel_size = (3, 3)
    sigma = (1.0, 1.0)
    res = _np_random_gaussian_blur(np, images, kernel_size, sigma)
    assert res.shape == (1, 5, 5, 3)


def test_random_sharpness() -> None:
    """Test random sharpness.

    Returns:
        None
    """
    images = np.ones((1, 5, 5, 3))
    factor = 1.5
    res = _np_random_sharpness(np, images, factor)
    assert res.shape == (1, 5, 5, 3)
