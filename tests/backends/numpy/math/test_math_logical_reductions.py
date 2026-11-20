"""Exhaustive tests for NumPy eager math logical reduction operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_logical_reductions as logical_reductions
from ml_switcheroo_compiler.backends.eager.reference_math.math_logical_reductions import _np_all, _np_count_nonzero
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import require_callable


def test_np_all_1d_cases() -> None:
    """Test _np_all with 1D arrays covering true, mixed, false, and empty inputs."""
    a1 = np.array([True, True, True])
    a2 = np.array([True, False, True])
    a3 = np.array([False, False])
    a4 = np.array([], dtype=bool)
    assert _np_all(np, a1)
    assert not _np_all(np, a2)
    assert not _np_all(np, a3)
    assert _np_all(np, a4)


def test_np_all_multidimensional_and_axis() -> None:
    """Test _np_all with multidimensional arrays along axes with and without keepdims."""
    arr: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ] = np.array([[True, True], [True, False]])

    # Full reduction
    assert not _np_all(np, arr)

    # Axis reduction
    reduced_axis0 = _np_all(np, arr, axis=0)
    assert reduced_axis0.tolist() == [True, False]

    reduced_axis1 = _np_all(np, arr, axis=1)
    assert reduced_axis1.tolist() == [True, False]

    # Keepdims
    reduced_keepdims = ensure_ndarray(_np_all(np, arr, axis=0, keepdims=True))
    assert reduced_keepdims.shape == (1, 2)
    assert reduced_keepdims.tolist() == [[True, False]]


def test_np_count_nonzero_1d_cases() -> None:
    """Test _np_count_nonzero with 1D arrays covering zero, mixed, non-zero, and empty."""
    assert _np_count_nonzero(np, np.array([0, 1, 0, 1])) == 2
    assert _np_count_nonzero(np, np.array([0, 0, 0])) == 0
    assert _np_count_nonzero(np, np.array([1, 2, 3])) == 3
    assert _np_count_nonzero(np, np.array([], dtype=int)) == 0


def test_np_count_nonzero_multidimensional_and_axis() -> None:
    """Test _np_count_nonzero with multidimensional arrays along axis."""
    arr: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ] = np.array([[0, 1, 2], [3, 0, 0]])

    assert _np_count_nonzero(np, arr) == 3

    # Along axis 0
    axis0_res = _np_count_nonzero(np, arr, axis=0)
    assert axis0_res.tolist() == [1, 1, 1]

    # Along axis 1
    axis1_res = _np_count_nonzero(np, arr, axis=1)
    assert axis1_res.tolist() == [2, 1]

    # Keepdims
    keepdims_res = _np_count_nonzero(np, arr, axis=1, keepdims=True)
    assert keepdims_res.shape == (2, 1)


def test_registry_integration() -> None:
    """Verify that All and CountNonzero are registered properly in numpy_eager_registry."""
    all_fn = numpy_eager_registry.get("All")
    assert all_fn is not None
    assert id(all_fn) == id(_np_all)
    assert all_fn(np, np.array([True, True]))

    count_fn = numpy_eager_registry.get("CountNonzero")
    assert count_fn is not None
    assert id(count_fn) == id(_np_count_nonzero)
    assert count_fn(np, np.array([1, 0, 2])) == 2


def test_math_logical_reductions() -> None:
    """Test All and CountNonzero reductions directly and via registry."""
    arr_all_true = np.array([True, True, True])
    arr_some_false = np.array([True, False, True])

    assert logical_reductions._np_all(np, arr_all_true)
    assert not logical_reductions._np_all(np, arr_some_false)

    reg_all = numpy_eager_registry.get("All")
    assert require_callable(reg_all)(np, arr_all_true)
    assert not require_callable(reg_all)(np, arr_some_false)

    arr_counts = np.array([[0, 1, 2], [3, 0, 0]])
    assert logical_reductions._np_count_nonzero(np, arr_counts) == 3

    reg_count = numpy_eager_registry.get("CountNonzero")
    assert require_callable(reg_count)(np, arr_counts) == 3
