"""Tests for numpy eager math reductions."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager.math_reductions import (
    _np_accumulate_n,
    _np_add_n,
    _np_any_op,
    _np_argmax,
    _np_argmin,
    _np_cumsum,
    _np_cumulative_logsumexp,
    _np_max,
    _np_mean,
    _np_min,
    _np_prod,
    _np_std,
    _np_sum,
    _np_variance,
)
from tests.testing_utils import ensure_callable, ensure_eager_tensor, ensure_ndarray

if typing.TYPE_CHECKING:
    pass


def test_np_sum() -> None:
    """Test function test_np_sum."""
    assert _np_sum(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == 6


def test_np_mean() -> None:
    """Test function test_np_mean."""
    assert _np_mean(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == 2.0


def test_np_max() -> None:
    """Test function test_np_max."""
    assert _np_max(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == 3


def test_np_min() -> None:
    """Test function test_np_min."""
    assert _np_min(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == 1


def test_np_variance() -> None:
    """Test function test_np_variance."""
    assert _np_variance(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == pytest.approx(0.666666666)  # ddof=0 default
    assert _np_variance(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3]), ddof=1) == 1.0


def test_np_std() -> None:
    """Test function test_np_std."""
    assert _np_std(ensure_eager_tensor(np), ensure_ndarray([1, 2, 3])) == pytest.approx(0.8164965)


def test_np_argmax() -> None:
    """Test function test_np_argmax."""
    assert _np_argmax(ensure_eager_tensor(np), ensure_ndarray([1, 3, 2])) == 1


def test_np_argmin() -> None:
    """Test function test_np_argmin."""
    assert _np_argmin(ensure_eager_tensor(np), ensure_ndarray([2, 1, 3])) == 1


def test_np_prod() -> None:
    """Test function test_np_prod."""
    assert _np_prod(ensure_eager_tensor(np), ensure_ndarray([2, 3, 4])) == 24


def test_np_any_op() -> None:
    """Test function test_np_any_op."""
    assert _np_any_op(ensure_eager_tensor(np), ensure_ndarray([False, True, False]))
    assert not _np_any_op(ensure_eager_tensor(np), ensure_ndarray([False, False]))


def test_np_cumsum() -> None:
    """Test function test_np_cumsum."""
    ensure_callable(np.testing.assert_allclose)(ensure_callable(_np_cumsum)(np, [1, 2, 3]), [1, 3, 6])


def test_np_add_n() -> None:
    """Test function test_np_add_n."""
    ensure_callable(np.testing.assert_allclose)(ensure_callable(_np_add_n)(np, [[1, 2], [3, 4], [5, 6]]), [9, 12])
    with pytest.raises(ValueError, match="inputs must not be empty"):
        _np_add_n(ensure_eager_tensor(np), [])


def test_np_accumulate_n() -> None:
    """Test function test_np_accumulate_n."""
    ensure_callable(np.testing.assert_allclose)(ensure_callable(_np_accumulate_n)(np, [[1, 2], [3, 4], [5, 6]]), [9, 12])
    with pytest.raises(ValueError, match="inputs must not be empty"):
        _np_accumulate_n(ensure_eager_tensor(np), [])


def test_np_cumulative_logsumexp() -> None:
    """Test function test_np_cumulative_logsumexp."""
    a = np.array([0.0, 1.0])
    res = _np_cumulative_logsumexp(ensure_eager_tensor(np), a, axis=0)
    expected = np.log(np.cumsum(np.exp(a), axis=0))
    ensure_callable(np.testing.assert_allclose)(res, expected)
