"""Tests for numpy eager vision filtering ops."""

from __future__ import annotations

import importlib
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_degeneration, _np_gaussian_blur, _np_median_filter, _np_sharpen
from tests.testing_utils import ensure_callable, ensure_eager_tensor

if typing.TYPE_CHECKING:
    pass
BlurConfig = ensure_eager_tensor(ensure_eager_tensor(importlib.import_module("ml_switcheroo_compiler.ops.vision.filtering")).BlurConfig)


def test_np_degeneration() -> None:
    """Test function test_np_degeneration."""
    res = _np_degeneration(ensure_eager_tensor(np), np.ones((2, 2)))
    ensure_callable(np.testing.assert_allclose)(res, ensure_callable(np.ones)((2, 2)))


def test_np_gaussian_blur() -> None:
    """Test function test_np_gaussian_blur."""
    with pytest.raises(ValueError):
        _np_gaussian_blur(ensure_eager_tensor(np), np.ones((2, 2)))


def test_np_median_filter() -> None:
    """Test function test_np_median_filter."""
    with pytest.raises(TypeError):
        _np_median_filter(ensure_eager_tensor(np), np.ones((2, 2)))


def test_np_sharpen() -> None:
    """Test function test_np_sharpen."""
    _np_sharpen(ensure_eager_tensor(np), np.ones((2, 2)))
    assert callable(test_np_sharpen)


def test_gaussian_blur_cov() -> None:
    """Test function test_gaussian_blur_cov."""
    config = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    ensure_callable(_np_gaussian_blur)(np, np.ones((1, 3, 3, 1)), config=config)
    assert config is not None
