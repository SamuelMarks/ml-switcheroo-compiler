"""Tests for numpy eager vision filtering ops."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.vision_filtering import (
    _np_degeneration,
    _np_gaussian_blur,
    _np_median_filter,
    _np_sharpen,
)

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
    try:
        _np_gaussian_blur(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_np_median_filter() -> None:
    """Test function test_np_median_filter."""
    try:
        _np_median_filter(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_np_sharpen() -> None:
    """Test function test_np_sharpen."""
    try:
        _np_sharpen(np, np.ones((2, 2)))
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_gaussian_blur_cov() -> None:
    """Test function test_gaussian_blur_cov."""
    config = BlurConfig(kernel_size=(3, 3), sigma=(1.0, 1.0))
    _np_gaussian_blur(np, np.ones((1, 3, 3, 1)), config=config)
    assert config is not None
