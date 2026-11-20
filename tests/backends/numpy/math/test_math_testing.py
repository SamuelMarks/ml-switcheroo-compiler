"""Behavioral unit tests for math testing."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import BackendArg, numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.backends.numpy.eager._math_advanced_test_ns import math_testing as test_module
from tests.testing_utils import (
    MockTestStub,
    NumpyTestArray,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
    unpack_pair,
)


class _TestMathTestingOpsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_testing_ops` split helpers."""

    _unused: int


def _test_math_testing_ops_part_1(_state: _TestMathTestingOpsState) -> None:
    """Execute part 1 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    pw_x = np.array([-2, -1, 0, 1, 2])
    pw_res = test_module._np_piecewise(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1])
    assert np.array_equal(ensure_ndarray(pw_res), [-1, -1, 1, 1, 1])
    reg_pw = numpy_eager_registry.get("Piecewise")
    assert np.array_equal(require_callable(reg_pw)(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1]), [-1, -1, 1, 1, 1])

    # PromoteTypes
    assert test_module._np_promotetypes(np, np.float32, np.int64) == np.dtype(np.float64)
    reg_promo = numpy_eager_registry.get("PromoteTypes")
    assert str(require_callable(reg_promo)(np, np.float32, np.int64)) == "float64"

    # ApplyAlongAxis
    arr_2d = np.array([[1, 2], [3, 4]])
    assert np.array_equal(ensure_ndarray(test_module._np_apply_along_axis(np, np.sum, 0, arr_2d)), [4, 6])
    reg_aaa = numpy_eager_registry.get("ApplyAlongAxis")
    assert np.array_equal(require_callable(reg_aaa)(np, np.sum, 0, arr_2d), [4, 6])

    # ArrayEquiv
    assert test_module._np_array_equiv_(np, [1, 2], [1, 2])
    assert not test_module._np_array_equiv_(np, [1, 2], [2, 3])
    reg_equiv = numpy_eager_registry.get("ArrayEquiv")
    assert require_callable(reg_equiv)(np, [1, 2], [1, 2])

    # BroadcastArrays
    b1, b2 = unpack_pair(test_module._np_broadcast_arrays_(np, [1, 2], [[1], [2]]))
    assert extract_test_shape(b1) == (2, 2)
    assert extract_test_shape(b2) == (2, 2)
    reg_bc = numpy_eager_registry.get("BroadcastArrays")
    rb1, rb2 = require_callable(reg_bc)(np, [1, 2], [[1], [2]])
    assert rb1.shape == (2, 2)


def _test_math_testing_ops_part_2(_state: _TestMathTestingOpsState) -> None:
    """Execute part 2 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    assert test_module._np_can_cast_(np, np.int32, np.int64)
    reg_cc = numpy_eager_registry.get("CanCast")
    assert require_callable(reg_cc)(np, np.int32, np.int64)

    class _HistStub(MockTestStub):
        """Stub backend module providing histogram routines."""

        @staticmethod
        def histogram(*args: BackendArg, **kw: BackendArg) -> tuple[NumpyTestArray, NumpyTestArray]:
            """Return mock 1D histogram counts and edges.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                tuple[NumpyTestArray, NumpyTestArray]: Counts and bin edges.
            """
            del args, kw
            return (np.array([2, 1]), np.array([0, 1, 2]))

        @staticmethod
        def histogram2d(*args: BackendArg, **kw: BackendArg) -> tuple[NumpyTestArray, NumpyTestArray, NumpyTestArray]:
            """Return mock 2D histogram counts and edges.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                tuple[NumpyTestArray, NumpyTestArray, NumpyTestArray]: 2D counts, x-edges, y-edges.
            """
            del args, kw
            return (np.zeros((2, 2)), np.array([0, 1, 2]), np.array([0, 1, 2]))

        @staticmethod
        def histogram_bin_edges(*args: BackendArg, **kw: BackendArg) -> NumpyTestArray:
            """Return mock histogram bin edges.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                NumpyTestArray: Bin edges array.
            """
            del args, kw
            return np.array([0, 1, 2])

        @staticmethod
        def histogramdd(*args: BackendArg, **kw: BackendArg) -> tuple[NumpyTestArray, list[NumpyTestArray]]:
            """Return mock N-D histogram counts and edges list.

            Args:
                *args: Positional arguments.
                **kw: Keyword arguments.

            Returns:
                tuple[NumpyTestArray, list[NumpyTestArray]]: Counts array and per-axis edges.
            """
            del args, kw
            return (np.zeros((2, 2)), [np.array([0, 1, 2]), np.array([0, 1, 2])])

    mock_hist_mod = _HistStub()
    h_data = np.array([1, 2, 1])
    counts, edges = unpack_pair(test_module._np_histogram_(mock_hist_mod, h_data, bins=2))
    del edges
    assert len(coerce_operand_list(counts)) == 2
    reg_hist = numpy_eager_registry.get("Histogram")
    assert len(require_callable(reg_hist)(mock_hist_mod, h_data, bins=2)[0]) == 2

    h2_out = test_module._np_histogram2d_(mock_hist_mod, [1, 2], [2, 3], bins=2)
    assert isinstance(h2_out, tuple)
    h2_counts, _, _ = h2_out
    assert extract_test_shape(h2_counts) == (2, 2)
    reg_h2d = numpy_eager_registry.get("Histogram2d")
    assert require_callable(reg_h2d)(mock_hist_mod, [1, 2], [2, 3], bins=2)[0].shape == (2, 2)

    bin_edges = test_module._np_histogram_bin_edges_(mock_hist_mod, [1, 2, 3], bins=2)
    assert len(coerce_operand_list(bin_edges)) == 3
    reg_edges = numpy_eager_registry.get("HistogramBinEdges")
    assert len(require_callable(reg_edges)(mock_hist_mod, [1, 2, 3], bins=2)) == 3

    hdd_counts, _ = unpack_pair(test_module._np_histogramdd_(mock_hist_mod, np.array([[1, 2], [3, 4]]), bins=2))
    assert extract_test_shape(hdd_counts) == (2, 2)
    reg_hdd = numpy_eager_registry.get("Histogramdd")
    assert require_callable(reg_hdd)(mock_hist_mod, np.array([[1, 2], [3, 4]]), bins=2)[0].shape == (2, 2)

    # Iscomplex and Iscomplexobj
    assert np.array_equal(ensure_ndarray(test_module._np_iscomplex_(np, [1 + 2j, 3])), [True, False])


def _test_math_testing_ops_part_3(_state: _TestMathTestingOpsState) -> None:
    """Execute part 3 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    assert test_module._np_iscomplexobj_(np, [1 + 2j])
    reg_is_c = numpy_eager_registry.get("Iscomplex")
    assert np.array_equal(require_callable(reg_is_c)(np, [1 + 2j, 3]), [True, False])
    reg_is_co = numpy_eager_registry.get("Iscomplexobj")
    assert require_callable(reg_is_co)(np, [1 + 2j])

    # Isdtype / Issubdtype
    assert test_module._np_issubdtype_op_(np, np.int32, np.integer)
    assert test_module._np_issubdtype_issubdtype_(np, np.float32, np.floating)
    reg_isdt = numpy_eager_registry.get("Isdtype")
    assert require_callable(reg_isdt)(np, np.int32, np.integer)
    reg_issub = numpy_eager_registry.get("Issubdtype")
    assert require_callable(reg_issub)(np, np.float32, np.floating)

    # Isreal and Isrealobj
    assert np.array_equal(ensure_ndarray(test_module._np_isreal_(np, [1 + 2j, 3])), [False, True])
    assert test_module._np_isrealobj_(np, [1.0, 2.0])
    reg_is_r = numpy_eager_registry.get("Isreal")
    assert np.array_equal(require_callable(reg_is_r)(np, [1 + 2j, 3]), [False, True])
    reg_is_ro = numpy_eager_registry.get("Isrealobj")
    assert require_callable(reg_is_ro)(np, [1.0, 2.0])

    # Isscalar
    assert test_module._np_isscalar_(np, 5.0)
    assert not test_module._np_isscalar_(np, [5.0])
    reg_isscalar = numpy_eager_registry.get("Isscalar")
    assert require_callable(reg_isscalar)(np, 5.0)

    # ResultType
    assert test_module._np_result_type_(np, 3, 1.5) == np.dtype(np.float64)


def _test_math_testing_ops_part_4(_state: _TestMathTestingOpsState) -> None:
    """Execute part 4 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    reg_rt = numpy_eager_registry.get("ResultType")
    assert str(require_callable(reg_rt)(np, 3, 1.5)) == "float64"

    # AssertOp branches
    assert np.array_equal(ensure_ndarray(test_module._np_assertop(np, [True, True])), [0.0])
    assert np.array_equal(ensure_ndarray(test_module._np_assertop(np, condition=[True])), [0.0])
    assert np.array_equal(ensure_ndarray(test_module._np_assertop(np)), [0.0])
    with pytest.raises(AssertionError):
        test_module._np_assertop(np, [True, False])

    reg_assertop = numpy_eager_registry.get("AssertOp")
    assert np.array_equal(require_callable(reg_assertop)(np, [True]), [0.0])


def test_math_testing_ops() -> None:
    """Test testing and utility operations directly and through eager registry.

    Returns:
        None
    """
    _state: _TestMathTestingOpsState = {}
    _test_math_testing_ops_part_1(_state)
    _test_math_testing_ops_part_2(_state)
    _test_math_testing_ops_part_3(_state)
    _test_math_testing_ops_part_4(_state)
