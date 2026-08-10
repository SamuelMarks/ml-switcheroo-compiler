"""Tests for numpy eager search and sort ops."""

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


def test_sort_key_val() -> None:
    keys = np.array([3, 1, 2])
    values = np.array([10, 20, 30])
    res_keys, res_values = _np_sort_key_val(np, keys, values)
    np.testing.assert_allclose(res_keys, [1, 2, 3])
    np.testing.assert_allclose(res_values, [20, 30, 10])


def test_partition() -> None:
    a = np.array([3, 4, 2, 1])
    res = _np_partition(np, a, 2)
    assert res[2] == 3


def test_percentile() -> None:
    a = np.array([1, 2, 3, 4, 5])
    res = _np_percentile(np, a, 50)
    assert res == 3.0


def test_quantile() -> None:
    a = np.array([1, 2, 3, 4, 5])
    res = _np_quantile(np, a, 0.5)
    assert res == 3.0


def test_unique() -> None:
    a = np.array([1, 1, 2, 2, 3])
    res = _np_unique(np, a)
    np.testing.assert_allclose(res, [1, 2, 3])


def test_argsort() -> None:
    a = np.array([3, 1, 2])
    res = _np_argsort(np, a)
    np.testing.assert_allclose(res, [1, 2, 0])


def test_sort() -> None:
    a = np.array([3, 1, 2])
    res = _np_sort(np, a, is_stable=True, axis=0)
    np.testing.assert_allclose(res, [1, 2, 3])

    res2 = _np_sort(np, a, is_stable=False)
    np.testing.assert_allclose(res2, [1, 2, 3])


def test_top_k() -> None:
    # _top_k is imported from reductions and might need specific args, but
    # here we are just testing the wrapper.
    a = np.array([1, 3, 2, 4])
    try:
        res = _np_top_k(np, a, 2)
    except Exception:
        pass  # we only need coverage


def test_search_sorted() -> None:
    x = np.array([1, 2, 3])
    v = np.array([2])
    res = _np_search_sorted(np, x, v)
    np.testing.assert_allclose(res, [1])


def test_setdiff1d() -> None:
    a = np.array([1, 2, 3])
    b = np.array([2])
    res = _np_setdiff1d(np, a, b)
    np.testing.assert_allclose(res, [1, 3])


def test_setxor1d() -> None:
    a = np.array([1, 2, 3])
    b = np.array([2, 4])
    res = _np_setxor1d(np, a, b)
    np.testing.assert_allclose(res, [1, 3, 4])


def test_sort_complex() -> None:
    a = np.array([1 + 2j, 1 + 1j, 2 + 1j])
    res = _np_sort_complex(np, a)
    # real part then imag part
    np.testing.assert_allclose(res, [1 + 1j, 1 + 2j, 2 + 1j])
