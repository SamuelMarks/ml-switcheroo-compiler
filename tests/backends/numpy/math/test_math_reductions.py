"""Behavioral unit tests for math reductions."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager.reference_math import reductions as red_module
from ml_switcheroo_compiler.backends.eager.reference_math.reductions import (
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
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import require_callable


def test_np_sum() -> None:
    """Test function test_np_sum."""
    assert _np_sum(np, ensure_ndarray([1, 2, 3])) == 6


def test_np_mean() -> None:
    """Test function test_np_mean."""
    assert _np_mean(np, ensure_ndarray([1, 2, 3])) == 2.0


def test_np_max() -> None:
    """Test function test_np_max."""
    assert _np_max(np, ensure_ndarray([1, 2, 3])) == 3


def test_np_min() -> None:
    """Test function test_np_min."""
    assert _np_min(np, ensure_ndarray([1, 2, 3])) == 1


def test_np_variance() -> None:
    """Test function test_np_variance."""
    assert _np_variance(np, ensure_ndarray([1, 2, 3])) == pytest.approx(0.666666666)  # ddof=0 default
    assert _np_variance(np, ensure_ndarray([1, 2, 3]), ddof=1) == 1.0


def test_np_std() -> None:
    """Test function test_np_std."""
    assert _np_std(np, ensure_ndarray([1, 2, 3])) == pytest.approx(0.8164965)


def test_np_argmax() -> None:
    """Test function test_np_argmax."""
    assert _np_argmax(np, ensure_ndarray([1, 3, 2])) == 1


def test_np_argmin() -> None:
    """Test function test_np_argmin."""
    assert _np_argmin(np, ensure_ndarray([2, 1, 3])) == 1


def test_np_prod() -> None:
    """Test function test_np_prod."""
    assert _np_prod(np, ensure_ndarray([2, 3, 4])) == 24


def test_np_any_op() -> None:
    """Test function test_np_any_op."""
    assert _np_any_op(np, ensure_ndarray([False, True, False]))
    assert not _np_any_op(np, ensure_ndarray([False, False]))


def test_np_cumsum() -> None:
    """Test function test_np_cumsum."""
    np.testing.assert_allclose(_np_cumsum(np, [1, 2, 3]), [1, 3, 6])


def test_np_add_n() -> None:
    """Test function test_np_add_n."""
    np.testing.assert_allclose(_np_add_n(np, [[1, 2], [3, 4], [5, 6]]), [9, 12])
    with pytest.raises(ValueError, match="inputs must not be empty"):
        _np_add_n(np, [])


def test_np_accumulate_n() -> None:
    """Test function test_np_accumulate_n."""
    np.testing.assert_allclose(_np_accumulate_n(np, [[1, 2], [3, 4], [5, 6]]), [9, 12])
    with pytest.raises(ValueError, match="inputs must not be empty"):
        _np_accumulate_n(np, [])


def test_np_cumulative_logsumexp() -> None:
    """Test function test_np_cumulative_logsumexp."""
    a = np.array([0.0, 1.0])
    res = _np_cumulative_logsumexp(np, a, axis=0)
    expected = np.log(np.cumsum(np.exp(a), axis=0))
    np.testing.assert_allclose(res, expected)


def test_math_reductions_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test math reduction operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        None
    """
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    from ml_switcheroo_compiler.backends.numpy.eager import distributed as dist_mod
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    _tcp_dist_ctx.world_size = 1

    # Pmean
    assert np.array_equal(ensure_ndarray(dist_mod._np_pmean(np, x, "axis0")), x)
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Pmean")(np, x, "axis0"), x)

    # Psum - world_size <= 1
    assert np.array_equal(dist_mod._np_psum(np, x), x)
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Psum")(np, x), x)

    # Psum - world_size > 1
    monkeypatch.setattr(_tcp_dist_ctx, "world_size", 2)
    monkeypatch.setattr(
        _tcp_dist_ctx,
        "all_reduce_ring",
        lambda tensor, op_type="sum", backend_module=None: tensor * 2.0,
    )
    psum_dist = dist_mod._np_psum(np, x)
    assert np.array_equal(psum_dist, x * 2.0)

    # ReducePrecision
    reduced = red_module._np_reduce_precision(np, x, 5, 10)
    assert reduced.dtype == np.float32
    assert np.allclose(reduced, x)
    reg_red_prec = numpy_eager_registry.get("ReducePrecision")
    assert np.allclose(require_callable(reg_red_prec)(np, x, 5, 10), x)

    # SparseReduceMax
    mat = np.array([[1, 5], [4, 2]])
    assert np.array_equal(red_module._np_sparsereducemax(np, mat), [5, 4])
    reg_sp_max = numpy_eager_registry.get("SparseReduceMax")
    assert reg_sp_max is not None
    assert reg_sp_max(np, mat) is not None
