"""Tests for test_math_testing_coverage."""

from __future__ import annotations

import types

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_testing as test_module


def test_math_testing_ops() -> None:
    """Test testing and utility operations directly and through eager registry.

    Returns:
        None
    """
    # Piecewise
    pw_x = np.array([-2, -1, 0, 1, 2])
    pw_res = test_module._np_piecewise(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1])
    assert np.array_equal(pw_res, [-1, -1, 1, 1, 1])
    reg_pw = numpy_eager_registry.get("Piecewise")
    assert np.array_equal(reg_pw(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1]), [-1, -1, 1, 1, 1])

    # PromoteTypes
    assert test_module._np_promotetypes(np, np.float32, np.int64) == np.dtype(np.float64)
    reg_promo = numpy_eager_registry.get("PromoteTypes")
    assert str(reg_promo(np, np.float32, np.int64)) == "float64"

    # ApplyAlongAxis
    arr_2d = np.array([[1, 2], [3, 4]])
    assert np.array_equal(test_module._np_apply_along_axis(np, np.sum, 0, arr_2d), [4, 6])
    reg_aaa = numpy_eager_registry.get("ApplyAlongAxis")
    assert np.array_equal(reg_aaa(np, np.sum, 0, arr_2d), [4, 6])

    # ArrayEquiv
    assert test_module._np_array_equiv_(np, [1, 2], [1, 2])
    assert not test_module._np_array_equiv_(np, [1, 2], [2, 3])
    reg_equiv = numpy_eager_registry.get("ArrayEquiv")
    assert reg_equiv(np, [1, 2], [1, 2])

    # BroadcastArrays
    b1, b2 = test_module._np_broadcast_arrays_(np, [1, 2], [[1], [2]])
    assert b1.shape == (2, 2)
    assert b2.shape == (2, 2)
    reg_bc = numpy_eager_registry.get("BroadcastArrays")
    rb1, rb2 = reg_bc(np, [1, 2], [[1], [2]])
    assert rb1.shape == (2, 2)

    # CanCast
    assert test_module._np_can_cast_(np, np.int32, np.int64)
    reg_cc = numpy_eager_registry.get("CanCast")
    assert reg_cc(np, np.int32, np.int64)

    # Histogram functions
    mock_hist_mod = types.SimpleNamespace(
        histogram=lambda *args, **kw: (np.array([2, 1]), np.array([0, 1, 2])),
        histogram2d=lambda *args, **kw: (np.zeros((2, 2)), np.array([0, 1, 2]), np.array([0, 1, 2])),
        histogram_bin_edges=lambda *args, **kw: np.array([0, 1, 2]),
        histogramdd=lambda *args, **kw: (np.zeros((2, 2)), [np.array([0, 1, 2]), np.array([0, 1, 2])]),
    )
    h_data = np.array([1, 2, 1])
    counts, edges = test_module._np_histogram_(mock_hist_mod, h_data, bins=2)
    assert len(counts) == 2
    reg_hist = numpy_eager_registry.get("Histogram")
    assert len(reg_hist(mock_hist_mod, h_data, bins=2)[0]) == 2

    h2_counts, _, _ = test_module._np_histogram2d_(mock_hist_mod, [1, 2], [2, 3], bins=2)
    assert h2_counts.shape == (2, 2)
    reg_h2d = numpy_eager_registry.get("Histogram2d")
    assert reg_h2d(mock_hist_mod, [1, 2], [2, 3], bins=2)[0].shape == (2, 2)

    bin_edges = test_module._np_histogram_bin_edges_(mock_hist_mod, [1, 2, 3], bins=2)
    assert len(bin_edges) == 3
    reg_edges = numpy_eager_registry.get("HistogramBinEdges")
    assert len(reg_edges(mock_hist_mod, [1, 2, 3], bins=2)) == 3

    hdd_counts, _ = test_module._np_histogramdd_(mock_hist_mod, np.array([[1, 2], [3, 4]]), bins=2)
    assert hdd_counts.shape == (2, 2)
    reg_hdd = numpy_eager_registry.get("Histogramdd")
    assert reg_hdd(mock_hist_mod, np.array([[1, 2], [3, 4]]), bins=2)[0].shape == (2, 2)

    # Iscomplex and Iscomplexobj
    assert np.array_equal(test_module._np_iscomplex_(np, [1 + 2j, 3]), [True, False])
    assert test_module._np_iscomplexobj_(np, [1 + 2j])
    reg_is_c = numpy_eager_registry.get("Iscomplex")
    assert np.array_equal(reg_is_c(np, [1 + 2j, 3]), [True, False])
    reg_is_co = numpy_eager_registry.get("Iscomplexobj")
    assert reg_is_co(np, [1 + 2j])

    # Isdtype / Issubdtype
    assert test_module._np_issubdtype_op_(np, np.int32, np.integer)
    assert test_module._np_issubdtype_issubdtype_(np, np.float32, np.floating)
    reg_isdt = numpy_eager_registry.get("Isdtype")
    assert reg_isdt(np, np.int32, np.integer)
    reg_issub = numpy_eager_registry.get("Issubdtype")
    assert reg_issub(np, np.float32, np.floating)

    # Isreal and Isrealobj
    assert np.array_equal(test_module._np_isreal_(np, [1 + 2j, 3]), [False, True])
    assert test_module._np_isrealobj_(np, [1.0, 2.0])
    reg_is_r = numpy_eager_registry.get("Isreal")
    assert np.array_equal(reg_is_r(np, [1 + 2j, 3]), [False, True])
    reg_is_ro = numpy_eager_registry.get("Isrealobj")
    assert reg_is_ro(np, [1.0, 2.0])

    # Isscalar
    assert test_module._np_isscalar_(np, 5.0)
    assert not test_module._np_isscalar_(np, [5.0])
    reg_isscalar = numpy_eager_registry.get("Isscalar")
    assert reg_isscalar(np, 5.0)

    # ResultType
    assert test_module._np_result_type_(np, 3, 1.5) == np.dtype(np.float64)
    reg_rt = numpy_eager_registry.get("ResultType")
    assert str(reg_rt(np, 3, 1.5)) == "float64"

    # AssertOp branches
    assert np.array_equal(test_module._np_assertop(np, [True, True]), [0.0])
    assert np.array_equal(test_module._np_assertop(np, condition=[True]), [0.0])
    assert np.array_equal(test_module._np_assertop(np), [0.0])
    with pytest.raises(AssertionError):
        test_module._np_assertop(np, [True, False])

    reg_assertop = numpy_eager_registry.get("AssertOp")
    assert np.array_equal(reg_assertop(np, [True]), [0.0])
