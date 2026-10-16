"""Tests for numpy eager math nan ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_nan import (
    _np_divide_no_nan,
    _np_multiply_no_nan,
    _np_nanmean,
    _np_nanmedian,
    _np_reciprocal_no_nan,
    _np_squared_difference,
    _np_xdivy,
    _np_xlog1py,
    _np_zero_fraction,
    _xlogy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_divide_no_nan() -> None:
    """Test divide_no_nan."""
    x = np.array([1.0, 1.0])
    y = np.array([1.0, 0.0])
    res = _np_divide_no_nan(np, x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1.0, 0.0])


def test_multiply_no_nan() -> None:
    """Test multiply_no_nan."""
    x = np.array([1.0, 1.0])
    y = np.array([1.0, 0.0])
    res = _np_multiply_no_nan(np, x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1.0, 0.0])


def test_squared_difference() -> None:
    """Test squared_difference."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 0.0])
    res = _np_squared_difference(np, x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [0.0, 4.0])


def test_xdivy() -> None:
    """Test xdivy."""
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 2.0])
    res = _np_xdivy(np, x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [0.0, 1.0])


def test_xlog1py() -> None:
    """Test xlog1py."""
    x = np.array([0.0, 1.0])
    y = np.array([1.0, np.e - 1])
    res = _np_xlog1py(np, x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [0.0, 1.0])


def test_reciprocal_no_nan() -> None:
    """Test reciprocal_no_nan."""
    x = np.array([0.0, 2.0])
    res = _np_reciprocal_no_nan(np, x)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [0.0, 0.5])


def test_zero_fraction() -> None:
    """Test zero_fraction."""
    x = np.array([0.0, 1.0, 2.0])
    res = _np_zero_fraction(np, x)
    assert res == 1.0 / 3.0

    # test empty array
    x_empty = np.array([])
    res_empty = _np_zero_fraction(np, x_empty)
    assert cast("Callable[..., EagerTensorProtocol]", np.isnan)(res_empty)


def test_xlogy() -> None:
    """Test xlogy."""
    x = np.array([0.0, 1.0])
    y = np.array([1.0, np.e])
    res = _xlogy(x, y)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [0.0, 1.0])


def test_nanmean() -> None:
    """Test nanmean."""
    a = np.array([1.0, np.nan, 3.0])
    res = _np_nanmean(np, a)
    assert res == 2.0


def test_nanmedian() -> None:
    """Test nanmedian."""
    a = np.array([1.0, np.nan, 3.0])
    res = _np_nanmedian(np, a)
    assert res == 2.0
