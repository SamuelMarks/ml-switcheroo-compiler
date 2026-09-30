"""Tests for test_math_reductions_coverage."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager.math_advanced import math_reductions as red_module


def test_math_reductions_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test math reduction operations directly and through eager registry.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        None
    """
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Pmean
    assert np.array_equal(red_module._np_pmean(np, x, "axis0"), x)
    assert np.array_equal(numpy_eager_registry.get("Pmean")(np, x, "axis0"), x)

    # Psum - world_size <= 1
    assert np.array_equal(red_module._np_psum(np, x), x)
    assert np.array_equal(numpy_eager_registry.get("Psum")(np, x), x)

    # Psum - world_size > 1
    from ml_switcheroo_compiler.backends.numpy.eager.distributed import _tcp_dist_ctx

    monkeypatch.setattr(_tcp_dist_ctx, "world_size", 2)
    monkeypatch.setattr(
        _tcp_dist_ctx,
        "all_reduce_ring",
        lambda tensor, op_type="sum", backend_module=None: tensor * 2.0,
    )
    psum_dist = red_module._np_psum(np, x)
    assert np.array_equal(psum_dist, x * 2.0)

    # ReducePrecision
    reduced = red_module._np_reduce_precision(np, x, 5, 10)
    assert reduced.dtype == np.float32
    assert np.allclose(reduced, x)
    reg_red_prec = numpy_eager_registry.get("ReducePrecision")
    assert np.allclose(reg_red_prec(np, x, 5, 10), x)

    # SparseReduceMax
    mat = np.array([[1, 5], [4, 2]])
    assert np.array_equal(red_module._np_sparsereducemax(np, mat), [5, 4])
    reg_sp_max = numpy_eager_registry.get("SparseReduceMax")
    assert reg_sp_max is not None
    assert reg_sp_max(np, mat) is not None
