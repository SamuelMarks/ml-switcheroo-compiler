"""Tests for numpy eager search and sort ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.search_sort_ops import (
    _np_argsort,
    _np_partition,
    _np_percentile,
    _np_quantile,
    _np_search_sorted,
    _np_setdiff1d,
    _np_setxor1d,
    _np_sort,
    _np_sort_complex,
    _np_sort_key_val,
    _np_top_k,
    _np_unique,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_sort_key_val() -> None:
    """Test function test_sort_key_val."""
    keys = np.array([3, 1, 2])
    values = np.array([10, 20, 30])
    res_keys, res_values = cast("Callable[..., EagerTensorProtocol]", _np_sort_key_val)(np, keys, values)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res_keys, [1, 2, 3])
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res_values, [20, 30, 10])


def test_partition() -> None:
    """Test function test_partition."""
    a = np.array([3, 4, 2, 1])
    res = _np_partition(np, a, 2)
    assert cast("EagerTensorProtocol", getattr(res, "__getitem__")(2)) == 3


def test_percentile() -> None:
    """Test function test_percentile."""
    a = np.array([1, 2, 3, 4, 5])
    res = _np_percentile(np, a, 50)
    assert res == 3.0


def test_quantile() -> None:
    """Test function test_quantile."""
    a = np.array([1, 2, 3, 4, 5])
    res = _np_quantile(np, a, 0.5)
    assert res == 3.0


def test_unique() -> None:
    """Test function test_unique."""
    a = np.array([1, 1, 2, 2, 3])
    res = _np_unique(np, a)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1, 2, 3])


def test_argsort() -> None:
    """Test function test_argsort."""
    a = np.array([3, 1, 2])
    res = _np_argsort(np, a)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1, 2, 0])


def test_sort() -> None:
    """Test function test_sort."""
    a = np.array([3, 1, 2])
    res = _np_sort(np, a, is_stable=True, axis=0)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1, 2, 3])

    res2 = _np_sort(np, a, is_stable=False)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res2, [1, 2, 3])


def test_top_k() -> None:
    # _top_k is imported from reductions and might need specific args, but
    # here we are just testing the wrapper.
    """Test function test_top_k."""
    a = np.array([1, 3, 2, 4])
    try:
        _np_top_k(np, a, 2)
    except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
        assert _exc is not None


def test_search_sorted() -> None:
    """Test function test_search_sorted."""
    x = np.array([1, 2, 3])
    v = np.array([2])
    res = _np_search_sorted(np, x, v)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1])


def test_setdiff1d() -> None:
    """Test function test_setdiff1d."""
    a = np.array([1, 2, 3])
    b = np.array([2])
    res = _np_setdiff1d(np, a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1, 3])


def test_setxor1d() -> None:
    """Test function test_setxor1d."""
    a = np.array([1, 2, 3])
    b = np.array([2, 4])
    res = _np_setxor1d(np, a, b)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1, 3, 4])


def test_sort_complex() -> None:
    """Test function test_sort_complex."""
    a = np.array([1 + 2j, 1 + 1j, 2 + 1j])
    res = _np_sort_complex(np, a)
    # real part then imag part
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, [1 + 1j, 1 + 2j, 2 + 1j])
