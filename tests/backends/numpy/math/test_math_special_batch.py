"""Numerical tests for NumPy eager special mathematical operations."""

from __future__ import annotations

import numpy as np

import ml_switcheroo_compiler.backends.numpy.eager.math_advanced as math_adv
from tests.testing_utils import ensure_eager_tensor


def test_math_misc_batch_1() -> None:
    """Verify numerical outputs of segment reductions, bit operations, and logsumexp."""
    np_mod = ensure_eager_tensor(np)
    seg = math_adv._np_segment_sum(np_mod, np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1]))
    assert np.allclose(np.asarray(seg), np.array([3.0, 7.0]))

    pop = math_adv._np_population_count(np_mod, np.array([0, 1, 3, 7], dtype=np.int32))
    assert np.array_equal(np.asarray(pop), np.array([0, 1, 2, 3]))

    lse = math_adv._np_logsumexp(np_mod, np.zeros((2, 2)), keepdims=False)
    assert np.isclose(float(np.asarray(lse)), float(np.log(4.0)))
