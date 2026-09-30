"""Exhaustive tests for NumPy eager math logical reduction operations."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_logical_reductions import (
    _np_all,
    _np_count_nonzero,
)


def test_np_all_1d_cases() -> None:
    """Test _np_all with 1D arrays covering true, mixed, false, and empty inputs."""
    a1 = np.array([True, True, True])
    a2 = np.array([True, False, True])
    a3 = np.array([False, False])
    a4 = np.array([], dtype=bool)
    assert _np_all(np, a1) == True
    assert _np_all(np, a2) == False
    assert _np_all(np, a3) == False
    assert _np_all(np, a4) == True


def test_np_all_multidimensional_and_axis() -> None:
    """Test _np_all with multidimensional arrays along axes with and without keepdims."""
    arr: np.ndarray = np.array([[True, True], [True, False]])

    # Full reduction
    assert _np_all(np, arr) == False

    # Axis reduction
    reduced_axis0 = _np_all(np, arr, axis=0)
    assert reduced_axis0.tolist() == [True, False]

    reduced_axis1 = _np_all(np, arr, axis=1)
    assert reduced_axis1.tolist() == [True, False]

    # Keepdims
    reduced_keepdims = _np_all(np, arr, axis=0, keepdims=True)
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
    arr: np.ndarray = np.array([[0, 1, 2], [3, 0, 0]])

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
    assert all_fn is _np_all
    assert all_fn(np, np.array([True, True])) == True

    count_fn = numpy_eager_registry.get("CountNonzero")
    assert count_fn is not None
    assert count_fn is _np_count_nonzero
    assert count_fn(np, np.array([1, 0, 2])) == 2
