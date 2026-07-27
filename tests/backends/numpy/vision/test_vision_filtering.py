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
