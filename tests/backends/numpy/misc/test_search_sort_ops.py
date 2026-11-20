"""Behavioral unit tests for search sort ops."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import search_sort_ops as sort_module
from ml_switcheroo_compiler.backends.eager.reference_math.search_sort_ops import (
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
from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import coerce_operand_list, make_mock_module, require_callable, unpack_pair


def test_sort_key_val() -> None:
    """Test function test_sort_key_val."""
    keys = np.array([3, 1, 2])
    values = np.array([10, 20, 30])
    (res_keys, res_values) = unpack_pair(_np_sort_key_val(np, keys, values))
    np.testing.assert_allclose(ensure_ndarray(res_keys), [1, 2, 3])
    np.testing.assert_allclose(ensure_ndarray(res_values), [20, 30, 10])


def test_partition() -> None:
    """Test function test_partition."""
    a = np.array([3, 4, 2, 1])
    res = _np_partition(np, a, 2)
    assert ensure_ndarray(res)[2] == 3


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
    np.testing.assert_allclose(ensure_ndarray(res), [1, 2, 3])


def test_argsort() -> None:
    """Test function test_argsort."""
    a = np.array([3, 1, 2])
    res = _np_argsort(np, a)
    np.testing.assert_allclose(ensure_ndarray(res), [1, 2, 0])


def test_sort() -> None:
    """Test function test_sort."""
    a = np.array([3, 1, 2])
    res = _np_sort(np, a, is_stable=True, axis=0)
    np.testing.assert_allclose(ensure_ndarray(res), [1, 2, 3])
    res2 = _np_sort(np, a, is_stable=False)
    np.testing.assert_allclose(ensure_ndarray(res2), [1, 2, 3])


def test_top_k() -> None:
    """Test function test_top_k."""
    a = np.array([1, 3, 2, 4])
    _out = _np_top_k(np, a, 2)
    assert len(coerce_operand_list(_out)) == 2


def test_search_sorted() -> None:
    """Test function test_search_sorted."""
    x = np.array([1, 2, 3])
    v = np.array([2])
    res = _np_search_sorted(np, x, v)
    np.testing.assert_allclose(ensure_ndarray(res), [1])


def test_setdiff1d() -> None:
    """Test function test_setdiff1d."""
    a = np.array([1, 2, 3])
    b = np.array([2])
    res = _np_setdiff1d(np, a, b)
    np.testing.assert_allclose(ensure_ndarray(res), [1, 3])


def test_setxor1d() -> None:
    """Test function test_setxor1d."""
    a = np.array([1, 2, 3])
    b = np.array([2, 4])
    res = _np_setxor1d(np, a, b)
    np.testing.assert_allclose(ensure_ndarray(res), [1, 3, 4])


def test_sort_complex() -> None:
    """Test function test_sort_complex."""
    a = np.array([1 + 2j, 1 + 1j, 2 + 1j])
    res = _np_sort_complex(np, a)
    np.testing.assert_allclose(ensure_ndarray(res), [1 + 1j, 1 + 2j, 2 + 1j])


def _mock_setdiff1d(*args: BackendArg, **kw: BackendArg) -> BackendArg:
    """Return mocked setdiff1d result.

    Args:
        *args: Positional arguments.
        **kw: Keyword arguments.

    Returns:
        BackendArg: Mocked difference array.
    """
    del args, kw
    return np.array([1, 3])


def _mock_setxor1d(*args: BackendArg, **kw: BackendArg) -> BackendArg:
    """Return mocked setxor1d result.

    Args:
        *args: Positional arguments.
        **kw: Keyword arguments.

    Returns:
        BackendArg: Mocked symmetric difference array.
    """
    del args, kw
    return np.array([1, 4])


def test_search_sort_ops() -> None:
    """Test search and sort operations directly and through eager registry.

    Returns:
        None
    """
    keys = np.array([3, 1, 2])
    values = np.array([30, 10, 20])

    # SortKeyVal
    sk, sv = unpack_pair(sort_module._np_sort_key_val(np, keys, values, axis=-1))
    assert np.array_equal(ensure_ndarray(sk), [1, 2, 3])
    assert np.array_equal(ensure_ndarray(sv), [10, 20, 30])
    reg_skv = numpy_eager_registry.get("SortKeyVal")
    rsk, rsv = require_callable(reg_skv)(np, keys, values, axis=-1)
    assert np.array_equal(rsk, [1, 2, 3])
    assert np.array_equal(rsv, [10, 20, 30])

    # Partition
    a = np.array([3, 4, 2, 1])
    part = sort_module._np_partition(np, a, 2)
    assert ensure_ndarray(part)[2] == 3
    reg_part = numpy_eager_registry.get("Partition")
    assert require_callable(reg_part)(np, a, 2)[2] == 3

    # Percentile
    data = np.array([1, 2, 3, 4, 5])
    assert sort_module._np_percentile(np, data, 50) == 3.0
    assert require_callable(numpy_eager_registry.get)("Percentile")(np, data, 50) == 3.0

    # Quantile
    data_float = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sort_module._np_quantile(np, data_float, 0.5) == 3.0
    assert require_callable(numpy_eager_registry.get)("Quantile")(np, data_float, 0.5) == 3.0

    # Unique
    u_data = np.array([1, 1, 2, 3, 2])
    assert np.array_equal(ensure_ndarray(sort_module._np_unique(np, u_data)), [1, 2, 3])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Unique")(np, u_data), [1, 2, 3])

    # ArgSort
    assert np.array_equal(ensure_ndarray(sort_module._np_argsort(np, keys)), [1, 2, 0])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("ArgSort")(np, keys), [1, 2, 0])

    # Sort with is_stable=True and is_stable=False
    assert np.array_equal(ensure_ndarray(sort_module._np_sort(np, keys, is_stable=True)), [1, 2, 3])
    assert np.array_equal(ensure_ndarray(sort_module._np_sort(np, keys, is_stable=False)), [1, 2, 3])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Sort")(np, keys, is_stable=True), [1, 2, 3])

    # TopK
    top_vals, top_idx = unpack_pair(sort_module._np_top_k(np, np.array([10, 50, 20, 40]), k=2))
    assert np.array_equal(ensure_ndarray(top_vals), [50, 40])
    reg_topk = numpy_eager_registry.get("TopK")
    rt_vals, _ = require_callable(reg_topk)(np, np.array([10, 50, 20, 40]), k=2)
    assert np.array_equal(rt_vals, [50, 40])

    # SearchSorted
    sorted_arr = np.array([1, 2, 3, 4])
    assert sort_module._np_search_sorted(np, sorted_arr, 2.5) == 2
    assert require_callable(numpy_eager_registry.get)("SearchSorted")(np, sorted_arr, 2.5) == 2

    # Setdiff1d and Setxor1d
    mock_set_mod = make_mock_module(
        setdiff1d=_mock_setdiff1d,
        setxor1d=_mock_setxor1d,
    )
    assert np.array_equal(ensure_ndarray(sort_module._np_setdiff1d(mock_set_mod, [1, 2, 3], [2, 4])), [1, 3])
    assert np.array_equal(
        require_callable(numpy_eager_registry.get)("Setdiff1d")(mock_set_mod, [1, 2, 3], [2, 4]), [1, 3]
    )

    assert np.array_equal(ensure_ndarray(sort_module._np_setxor1d(mock_set_mod, [1, 2, 3], [2, 3, 4])), [1, 4])
    assert np.array_equal(
        require_callable(numpy_eager_registry.get)("Setxor1d")(mock_set_mod, [1, 2, 3], [2, 3, 4]), [1, 4]
    )

    # SortComplex
    c_arr = np.array([2 + 3j, 1 - 1j, 1 + 2j])
    assert np.array_equal(ensure_ndarray(sort_module._np_sort_complex(np, c_arr)), [1 - 1j, 1 + 2j, 2 + 3j])
    assert np.array_equal(
        require_callable(numpy_eager_registry.get)("SortComplex")(np, c_arr), [1 - 1j, 1 + 2j, 2 + 3j]
    )
