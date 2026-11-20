"""Tests for test_sparse."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray

if typing.TYPE_CHECKING:
    pass


def test_sparse_ops_coverage() -> None:
    """Test function test_sparse_ops_coverage."""
    from tests.backends.numpy.eager._math_advanced_test_ns import (
        _np_raggeddot,
        _np_sparsemapvalues,
        _np_sparsereducemax,
        _np_sparsereshape,
        _np_sparsesampledadd,
        _np_sparsesegmentsum,
        _np_sparsetranspose,
    )

    a = np.array([[1, 2], [3, 4]])
    b = np.array([[1, 0], [0, 1]])

    assert np.array_equal(ensure_ndarray(_np_raggeddot(np, a, b)), a)
    assert np.array_equal(ensure_ndarray(_np_sparsemapvalues(np, lambda x: x * 2, a)), a * 2)
    assert np.array_equal(_np_sparsereducemax(np, a), np.array([2, 4]))
    assert np.array_equal(ensure_ndarray(_np_sparsereshape(np, a, (4,))), np.array([1, 2, 3, 4]))
    assert np.array_equal(ensure_ndarray(_np_sparsesampledadd(np, a, b)), a + b)
    assert np.array_equal(
        _np_sparsesegmentsum(np, a, np.array([0, 1]), np.array([0, 0])),
        np.array([[4, 6]]),
    )
    assert np.array_equal(ensure_ndarray(_np_sparsetranspose(np, a)), a.T)
