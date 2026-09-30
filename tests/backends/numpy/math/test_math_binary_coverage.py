"""Tests for test_math_binary_coverage."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_binary as bin_module


def test_math_binary_ops() -> None:
    """Test binary operations directly and through eager registry.

    Returns:
        None
    """
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])

    assert np.allclose(bin_module._np_add(np, a, b), [3.0, 6.0])
    assert np.allclose(numpy_eager_registry.get("Add")(np, a, b), [3.0, 6.0])

    assert np.allclose(bin_module._np_subtract(np, a, b), [1.0, 2.0])
    assert np.allclose(numpy_eager_registry.get("Subtract")(np, a, b), [1.0, 2.0])

    assert np.allclose(bin_module._np_multiply(np, a, b), [2.0, 8.0])
    assert np.allclose(numpy_eager_registry.get("Multiply")(np, a, b), [2.0, 8.0])

    assert np.allclose(bin_module._np_true_divide(np, a, b), [2.0, 2.0])
    assert np.allclose(numpy_eager_registry.get("TrueDivide")(np, a, b), [2.0, 2.0])

    assert np.allclose(bin_module._np_maximum(np, a, b), [2.0, 4.0])
    assert np.allclose(numpy_eager_registry.get("Maximum")(np, a, b), [2.0, 4.0])

    assert np.allclose(bin_module._np_minimum(np, a, b), [1.0, 2.0])
    assert np.allclose(numpy_eager_registry.get("Minimum")(np, a, b), [1.0, 2.0])

    i_a = np.array([1, 3], dtype=np.int32)
    i_b = np.array([2, 3], dtype=np.int32)

    assert np.array_equal(bin_module._np_bitwise_and(np, i_a, i_b), [0, 3])
    assert np.array_equal(numpy_eager_registry.get("BitwiseAnd")(np, i_a, i_b), [0, 3])

    assert np.array_equal(bin_module._np_bitwise_or(np, i_a, i_b), [3, 3])
    assert np.array_equal(numpy_eager_registry.get("BitwiseOr")(np, i_a, i_b), [3, 3])

    assert np.array_equal(bin_module._np_bitwise_xor(np, i_a, i_b), [3, 0])
    assert np.array_equal(numpy_eager_registry.get("BitwiseXor")(np, i_a, i_b), [3, 0])

    assert np.array_equal(bin_module._np_left_shift(np, i_a, 1), [2, 6])
    assert np.array_equal(numpy_eager_registry.get("LeftShift")(np, i_a, 1), [2, 6])

    assert np.array_equal(bin_module._np_right_shift(np, i_a, 1), [0, 1])
    assert np.array_equal(numpy_eager_registry.get("RightShift")(np, i_a, 1), [0, 1])

    assert np.allclose(bin_module._np_logaddexp(np, a, b), np.logaddexp(a, b))
    assert np.allclose(numpy_eager_registry.get("Logaddexp")(np, a, b), np.logaddexp(a, b))

    assert np.allclose(bin_module._np_logaddexp2(np, a, b), np.logaddexp2(a, b))
    assert np.allclose(numpy_eager_registry.get("Logaddexp2")(np, a, b), np.logaddexp2(a, b))

    nan_arr = np.array([np.nan, 1.0])
    assert np.allclose(bin_module._np_nan_to_num(np, nan_arr), [0.0, 1.0])
    assert np.allclose(numpy_eager_registry.get("NanToNum")(np, nan_arr), [0.0, 1.0])

    mant, exp = bin_module._np_frexp(np, a)
    assert len(mant) == 2
    r_mant, _ = numpy_eager_registry.get("Frexp")(np, a)
    assert len(r_mant) == 2

    assert np.allclose(bin_module._np_clip(np, a, 1.5, 3.0), [2.0, 3.0])
    assert np.allclose(numpy_eager_registry.get("Clip")(np, a, 1.5, 3.0), [2.0, 3.0])

    assert bin_module._np_amax(np, a) == 4.0
    assert numpy_eager_registry.get("Amax")(np, a) == 4.0

    assert bin_module._np_amin(np, a) == 2.0
    assert numpy_eager_registry.get("Amin")(np, a) == 2.0

    prob = np.array([0.5, 0.8])
    assert np.allclose(bin_module._np_logit(np, prob), np.log(prob / (1.0 - prob)))
    assert np.allclose(numpy_eager_registry.get("Logit")(np, prob), np.log(prob / (1.0 - prob)))

    # Polygamma branches
    assert bin_module._np_polygamma(np, 0, 1.0) is not None
    assert bin_module._np_polygamma(np, 0, x=1.0) is not None
    assert np.array_equal(bin_module._np_polygamma(np, np.array([0])), [0])
    assert numpy_eager_registry.get("Polygamma")(np, 0, 1.0) is not None

    # Zeta branches
    assert bin_module._np_zeta(np, 2.0, 1.0) is not None
    assert bin_module._np_zeta(np, 2.0, q=1.0) is not None
    assert np.array_equal(bin_module._np_zeta(np, np.array([2.0])), [0.0])
    assert numpy_eager_registry.get("Zeta")(np, 2.0, 1.0) is not None

    # Remainder
    assert np.allclose(bin_module._eager_remainder(np, 7, 3), 1)
    assert np.allclose(numpy_eager_registry.get("Remainder")(np, 7, 3), 1)
