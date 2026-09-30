"""Tests for test_search_sort_coverage."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import search_sort_ops as sort_module


def test_search_sort_ops() -> None:
    """Test search and sort operations directly and through eager registry.

    Returns:
        None
    """
    keys = np.array([3, 1, 2])
    values = np.array([30, 10, 20])

    # SortKeyVal
    sk, sv = sort_module._np_sort_key_val(np, keys, values, axis=-1)
    assert np.array_equal(sk, [1, 2, 3])
    assert np.array_equal(sv, [10, 20, 30])
    reg_skv = numpy_eager_registry.get("SortKeyVal")
    rsk, rsv = reg_skv(np, keys, values, axis=-1)
    assert np.array_equal(rsk, [1, 2, 3])
    assert np.array_equal(rsv, [10, 20, 30])

    # Partition
    a = np.array([3, 4, 2, 1])
    part = sort_module._np_partition(np, a, 2)
    assert part[2] == 3
    reg_part = numpy_eager_registry.get("Partition")
    assert reg_part(np, a, 2)[2] == 3

    # Percentile
    data = np.array([1, 2, 3, 4, 5])
    assert sort_module._np_percentile(np, data, 50) == 3.0
    assert numpy_eager_registry.get("Percentile")(np, data, 50) == 3.0

    # Quantile
    data_float = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sort_module._np_quantile(np, data_float, 0.5) == 3.0
    assert numpy_eager_registry.get("Quantile")(np, data_float, 0.5) == 3.0

    # Unique
    u_data = np.array([1, 1, 2, 3, 2])
    assert np.array_equal(sort_module._np_unique(np, u_data), [1, 2, 3])
    assert np.array_equal(numpy_eager_registry.get("Unique")(np, u_data), [1, 2, 3])

    # ArgSort
    assert np.array_equal(sort_module._np_argsort(np, keys), [1, 2, 0])
    assert np.array_equal(numpy_eager_registry.get("ArgSort")(np, keys), [1, 2, 0])

    # Sort with is_stable=True and is_stable=False
    assert np.array_equal(sort_module._np_sort(np, keys, is_stable=True), [1, 2, 3])
    assert np.array_equal(sort_module._np_sort(np, keys, is_stable=False), [1, 2, 3])
    assert np.array_equal(numpy_eager_registry.get("Sort")(np, keys, is_stable=True), [1, 2, 3])

    # TopK
    top_vals, top_idx = sort_module._np_top_k(np, np.array([10, 50, 20, 40]), k=2)
    assert np.array_equal(top_vals, [50, 40])
    reg_topk = numpy_eager_registry.get("TopK")
    rt_vals, _ = reg_topk(np, np.array([10, 50, 20, 40]), k=2)
    assert np.array_equal(rt_vals, [50, 40])

    # SearchSorted
    sorted_arr = np.array([1, 2, 3, 4])
    assert sort_module._np_search_sorted(np, sorted_arr, 2.5) == 2
    assert numpy_eager_registry.get("SearchSorted")(np, sorted_arr, 2.5) == 2

    # Setdiff1d and Setxor1d
    mock_set_mod = types.SimpleNamespace(
        setdiff1d=lambda *args, **kw: np.array([1, 3]),
        setxor1d=lambda *args, **kw: np.array([1, 4]),
    )
    assert np.array_equal(sort_module._np_setdiff1d(mock_set_mod, [1, 2, 3], [2, 4]), [1, 3])
    assert np.array_equal(numpy_eager_registry.get("Setdiff1d")(mock_set_mod, [1, 2, 3], [2, 4]), [1, 3])

    assert np.array_equal(sort_module._np_setxor1d(mock_set_mod, [1, 2, 3], [2, 3, 4]), [1, 4])
    assert np.array_equal(numpy_eager_registry.get("Setxor1d")(mock_set_mod, [1, 2, 3], [2, 3, 4]), [1, 4])

    # SortComplex
    c_arr = np.array([2 + 3j, 1 - 1j, 1 + 2j])
    assert np.array_equal(sort_module._np_sort_complex(np, c_arr), [1 - 1j, 1 + 2j, 2 + 3j])
    assert np.array_equal(numpy_eager_registry.get("SortComplex")(np, c_arr), [1 - 1j, 1 + 2j, 2 + 3j])
