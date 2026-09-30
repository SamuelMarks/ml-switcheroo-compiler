"""Tests for test_math_reductions_advanced."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_reductions as reductions_mod


def test_math_reductions() -> None:
    """Test all reduction functions in math_reductions.

    Returns:
        None
    """
    arr = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(reductions_mod._np_sum(np, arr), 10.0)
    assert np.isclose(numpy_eager_registry.get("Sum")(np, arr), 10.0)

    assert np.isclose(reductions_mod._np_mean(np, arr), 2.5)
    assert np.isclose(numpy_eager_registry.get("Mean")(np, arr), 2.5)

    assert np.isclose(reductions_mod._np_max(np, arr), 4.0)
    assert np.isclose(numpy_eager_registry.get("Max")(np, arr), 4.0)

    assert np.isclose(reductions_mod._np_min(np, arr), 1.0)
    assert np.isclose(numpy_eager_registry.get("Min")(np, arr), 1.0)

    assert np.isclose(reductions_mod._np_variance(np, arr), np.var(arr))
    assert np.isclose(numpy_eager_registry.get("Variance")(np, arr), np.var(arr))

    assert np.isclose(reductions_mod._np_std(np, arr), np.std(arr))
    assert np.isclose(numpy_eager_registry.get("Std")(np, arr), np.std(arr))

    assert reductions_mod._np_argmax(np, arr) == 3
    assert numpy_eager_registry.get("Argmax")(np, arr) == 3

    assert reductions_mod._np_argmin(np, arr) == 0
    assert numpy_eager_registry.get("Argmin")(np, arr) == 0

    assert np.isclose(reductions_mod._np_prod(np, arr), 24.0)
    assert np.isclose(numpy_eager_registry.get("Prod")(np, arr), 24.0)

    assert reductions_mod._np_any_op(np, np.array([False, True]))
    assert numpy_eager_registry.get("AnyOp")(np, np.array([False, True]))

    assert np.array_equal(reductions_mod._np_cumsum(np, arr), np.cumsum(arr))
    assert np.array_equal(numpy_eager_registry.get("Cumsum")(np, arr), np.cumsum(arr))

    # AddN & AccumulateN
    with pytest.raises(ValueError, match="inputs must not be empty"):
        reductions_mod._np_add_n(np, [])
    with pytest.raises(ValueError, match="inputs must not be empty"):
        reductions_mod._np_accumulate_n(np, [])

    single_input = [np.array([1, 2])]
    assert np.array_equal(reductions_mod._np_add_n(np, single_input), [1, 2])
    assert np.array_equal(reductions_mod._np_accumulate_n(np, single_input), [1, 2])

    multi_inputs = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    assert np.array_equal(reductions_mod._np_add_n(np, multi_inputs), [9, 12])
    assert np.array_equal(numpy_eager_registry.get("AddN")(np, multi_inputs), [9, 12])

    assert np.array_equal(reductions_mod._np_accumulate_n(np, multi_inputs), [9, 12])
    assert np.array_equal(numpy_eager_registry.get("AccumulateN")(np, multi_inputs), [9, 12])

    # CumulativeLogsumexp
    c_lse = reductions_mod._np_cumulative_logsumexp(np, arr)
    assert len(c_lse) == 4
    assert len(numpy_eager_registry.get("CumulativeLogsumexp")(np, arr)) == 4
