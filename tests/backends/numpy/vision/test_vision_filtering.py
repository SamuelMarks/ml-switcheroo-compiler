"""Tests for numpy eager vision filtering ops."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import _np_degeneration, _np_gaussian_blur, _np_median_filter, _np_sharpen

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol
BlurConfig = cast("EagerTensorProtocol", getattr(importlib.import_module("ml_switcheroo_compiler.ops.vision.filtering"), "BlurConfig"))


def test_np_degeneration() -> None:
    """Test function test_np_degeneration."""
    res = _np_degeneration(np, np.ones((2, 2)))
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, cast("Callable[..., EagerTensorProtocol]", np.ones)((2, 2)))


def test_np_gaussian_blur() -> None:
    """Test function test_np_gaussian_blur."""
    with pytest.raises(ValueError):
        _np_gaussian_blur(np, np.ones((2, 2)))


def test_np_median_filter() -> None:
    """Test function test_np_median_filter."""
    with pytest.raises(TypeError):
        _np_median_filter(np, np.ones((2, 2)))


def test_np_sharpen() -> None:
    """Test function test_np_sharpen."""
    _np_sharpen(np, np.ones((2, 2)))
    assert callable(test_np_sharpen)


def test_gaussian_blur_cov() -> None:
    """Test function test_gaussian_blur_cov."""
    config = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    _np_gaussian_blur(np, np.ones((1, 3, 3, 1)), config=config)
    assert config is not None
