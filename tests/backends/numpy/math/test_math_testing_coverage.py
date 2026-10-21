"""Tests for test_math_testing_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_testing as test_module
from tests.testing_utils import as_callable, as_mod, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_math_testing_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    pw_x = np.array([-2, -1, 0, 1, 2])
    pw_res = test_module._np_piecewise(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1])
    assert as_callable(np.array_equal)(pw_res, [-1, -1, 1, 1, 1])
    reg_pw = numpy_eager_registry.get("Piecewise")
    assert as_callable(np.array_equal)(as_callable(reg_pw)(np, pw_x, [pw_x < 0, pw_x >= 0], [-1, 1]), [-1, -1, 1, 1, 1])

    # PromoteTypes
    assert test_module._np_promotetypes(np, np.float32, np.int64) == np.dtype(np.float64)
    reg_promo = numpy_eager_registry.get("PromoteTypes")
    assert as_callable(str)(as_callable(reg_promo)(np, np.float32, np.int64)) == "float64"

    # ApplyAlongAxis
    arr_2d = np.array([[1, 2], [3, 4]])
    assert as_callable(np.array_equal)(as_callable(test_module._np_apply_along_axis)(np, np.sum, 0, arr_2d), [4, 6])
    reg_aaa = numpy_eager_registry.get("ApplyAlongAxis")
    assert as_callable(np.array_equal)(as_callable(reg_aaa)(np, np.sum, 0, arr_2d), [4, 6])

    # ArrayEquiv
    assert test_module._np_array_equiv_(np, [1, 2], [1, 2])
    assert not test_module._np_array_equiv_(np, [1, 2], [2, 3])
    reg_equiv = numpy_eager_registry.get("ArrayEquiv")
    assert as_callable(reg_equiv)(np, [1, 2], [1, 2])

    # BroadcastArrays
    b1, b2 = as_callable(test_module._np_broadcast_arrays_)(np, [1, 2], [[1], [2]])
    assert as_tensor(getattr(b1, "shape")) == (2, 2)
    assert as_tensor(getattr(b2, "shape")) == (2, 2)
    reg_bc = numpy_eager_registry.get("BroadcastArrays")
    rb1, rb2 = as_callable(reg_bc)(np, [1, 2], [[1], [2]])
    assert as_tensor(getattr(rb1, "shape")) == (2, 2)
    getattr(_state, "update")({"arr_2d": arr_2d, "b1": b1, "b2": b2, "pw_res": pw_res})
    getattr(_state, "update")({"pw_x": pw_x, "rb1": rb1, "rb2": rb2, "reg_aaa": reg_aaa})
    getattr(_state, "update")({"reg_bc": reg_bc, "reg_equiv": reg_equiv, "reg_promo": reg_promo, "reg_pw": reg_pw})
    _state.update(locals())


def _test_math_testing_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    assert test_module._np_can_cast_(np, np.int32, np.int64)
    reg_cc = numpy_eager_registry.get("CanCast")
    assert as_callable(reg_cc)(np, np.int32, np.int64)
    mock_hist_mod = types.SimpleNamespace(
        histogram=lambda *args, **kw: (np.array([2, 1]), np.array([0, 1, 2])),
        histogram2d=lambda *args, **kw: (np.zeros((2, 2)), np.array([0, 1, 2]), np.array([0, 1, 2])),
        histogram_bin_edges=lambda *args, **kw: np.array([0, 1, 2]),
        histogramdd=lambda *args, **kw: (np.zeros((2, 2)), [np.array([0, 1, 2]), np.array([0, 1, 2])]),
    )
    h_data = np.array([1, 2, 1])
    counts, edges = as_callable(test_module._np_histogram_)(mock_hist_mod, h_data, bins=2)
    assert as_callable(len)(counts) == 2
    reg_hist = numpy_eager_registry.get("Histogram")
    assert as_callable(len)(as_tensor(getattr(as_callable(reg_hist)(mock_hist_mod, h_data, bins=2), "__getitem__")(0))) == 2

    h2_counts, _, _ = as_callable(test_module._np_histogram2d_)(mock_hist_mod, [1, 2], [2, 3], bins=2)
    assert as_tensor(getattr(h2_counts, "shape")) == (2, 2)
    reg_h2d = numpy_eager_registry.get("Histogram2d")
    assert as_tensor(getattr(as_tensor(getattr(as_callable(reg_h2d)(mock_hist_mod, [1, 2], [2, 3], bins=2), "__getitem__")(0)), "shape")) == (2, 2)

    bin_edges = test_module._np_histogram_bin_edges_(as_mod(mock_hist_mod), [1, 2, 3], bins=2)
    assert as_callable(len)(bin_edges) == 3
    reg_edges = numpy_eager_registry.get("HistogramBinEdges")
    assert as_callable(len)(as_callable(reg_edges)(mock_hist_mod, [1, 2, 3], bins=2)) == 3

    hdd_counts, _ = as_callable(test_module._np_histogramdd_)(mock_hist_mod, as_callable(np.array)([[1, 2], [3, 4]]), bins=2)
    assert as_tensor(getattr(hdd_counts, "shape")) == (2, 2)
    reg_hdd = numpy_eager_registry.get("Histogramdd")
    assert as_tensor(getattr(as_tensor(getattr(as_callable(reg_hdd)(mock_hist_mod, as_callable(np.array)([[1, 2], [3, 4]]), bins=2), "__getitem__")(0)), "shape")) == (2, 2)

    # Iscomplex and Iscomplexobj
    assert as_callable(np.array_equal)(as_callable(test_module._np_iscomplex_)(np, [1 + 2j, 3]), [True, False])
    getattr(_state, "update")({"_": _, "bin_edges": bin_edges, "counts": counts, "edges": edges})
    getattr(_state, "update")({"h2_counts": h2_counts, "h_data": h_data, "hdd_counts": hdd_counts, "mock_hist_mod": mock_hist_mod})
    getattr(_state, "update")({"reg_cc": reg_cc, "reg_edges": reg_edges, "reg_h2d": reg_h2d, "reg_hdd": reg_hdd})
    getattr(_state, "update")({"reg_hist": reg_hist})
    _state.update(locals())


def _test_math_testing_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    assert test_module._np_iscomplexobj_(np, [1 + 2j])
    reg_is_c = numpy_eager_registry.get("Iscomplex")
    assert as_callable(np.array_equal)(as_callable(reg_is_c)(np, [1 + 2j, 3]), [True, False])
    reg_is_co = numpy_eager_registry.get("Iscomplexobj")
    assert as_callable(reg_is_co)(np, [1 + 2j])

    # Isdtype / Issubdtype
    assert test_module._np_issubdtype_op_(np, np.int32, np.integer)
    assert test_module._np_issubdtype_issubdtype_(np, np.float32, np.floating)
    reg_isdt = numpy_eager_registry.get("Isdtype")
    assert as_callable(reg_isdt)(np, np.int32, np.integer)
    reg_issub = numpy_eager_registry.get("Issubdtype")
    assert as_callable(reg_issub)(np, np.float32, np.floating)

    # Isreal and Isrealobj
    assert as_callable(np.array_equal)(as_callable(test_module._np_isreal_)(np, [1 + 2j, 3]), [False, True])
    assert test_module._np_isrealobj_(np, [1.0, 2.0])
    reg_is_r = numpy_eager_registry.get("Isreal")
    assert as_callable(np.array_equal)(as_callable(reg_is_r)(np, [1 + 2j, 3]), [False, True])
    reg_is_ro = numpy_eager_registry.get("Isrealobj")
    assert as_callable(reg_is_ro)(np, [1.0, 2.0])

    # Isscalar
    assert test_module._np_isscalar_(np, 5.0)
    assert not test_module._np_isscalar_(np, [5.0])
    reg_isscalar = numpy_eager_registry.get("Isscalar")
    assert as_callable(reg_isscalar)(np, 5.0)

    # ResultType
    assert test_module._np_result_type_(np, 3, 1.5) == np.dtype(np.float64)
    getattr(_state, "update")({"reg_is_c": reg_is_c, "reg_is_co": reg_is_co, "reg_is_r": reg_is_r, "reg_is_ro": reg_is_ro})
    getattr(_state, "update")({"reg_isdt": reg_isdt, "reg_isscalar": reg_isscalar, "reg_issub": reg_issub})
    _state.update(locals())


def _test_math_testing_ops_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 4 of test_math_testing_ops.

    Args:
        _state: Shared test state dictionary.
    """
    reg_rt = numpy_eager_registry.get("ResultType")
    assert as_callable(str)(as_callable(reg_rt)(np, 3, 1.5)) == "float64"

    # AssertOp branches
    assert as_callable(np.array_equal)(as_callable(test_module._np_assertop)(np, [True, True]), [0.0])
    assert as_callable(np.array_equal)(as_callable(test_module._np_assertop)(np, condition=[True]), [0.0])
    assert as_callable(np.array_equal)(as_callable(test_module._np_assertop)(np), [0.0])
    with pytest.raises(AssertionError):
        test_module._np_assertop(np, [True, False])

    reg_assertop = numpy_eager_registry.get("AssertOp")
    assert as_callable(np.array_equal)(as_callable(reg_assertop)(np, [True]), [0.0])
    getattr(_state, "update")({"reg_assertop": reg_assertop, "reg_rt": reg_rt})


def test_math_testing_ops() -> None:
    """Test testing and utility operations directly and through eager registry.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_testing_ops_part_1(_state)
    _test_math_testing_ops_part_2(_state)
    _test_math_testing_ops_part_3(_state)
    _test_math_testing_ops_part_4(_state)
    assert len(_state) > 0
