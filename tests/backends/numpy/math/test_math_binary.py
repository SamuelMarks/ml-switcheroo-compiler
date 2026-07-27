"""Tests for numpy eager math binary ops."""

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager.math_binary import (
    _eager_remainder,
    _np_add,
    _np_amax,
    _np_amin,
    _np_bitwise_and,
    _np_bitwise_or,
    _np_bitwise_xor,
    _np_clip,
    _np_frexp,
    _np_left_shift,
    _np_logaddexp,
    _np_logaddexp2,
    _np_logit,
    _np_maximum,
    _np_minimum,
    _np_multiply,
    _np_nan_to_num,
    _np_polygamma,
    _np_right_shift,
    _np_subtract,
    _np_true_divide,
    _np_zeta,
)


def test_math_binary_ops() -> None:
    """Test math binary ops."""
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])
    ia = np.array([2, 4])
    ib = np.array([1, 2])

    assert np.array_equal(_np_add(np, a, b), a + b)
    assert np.array_equal(_np_subtract(np, a, b), a - b)
    assert np.array_equal(_np_multiply(np, a, b), a * b)
    assert np.array_equal(_np_true_divide(np, a, b), a / b)
    assert np.array_equal(_np_maximum(np, a, b), np.maximum(a, b))
    assert np.array_equal(_np_minimum(np, a, b), np.minimum(a, b))

    assert np.array_equal(_np_bitwise_and(np, ia, ib), np.bitwise_and(ia, ib))
    assert np.array_equal(_np_bitwise_or(np, ia, ib), np.bitwise_or(ia, ib))
    assert np.array_equal(_np_bitwise_xor(np, ia, ib), np.bitwise_xor(ia, ib))
    assert np.array_equal(_np_left_shift(np, ia, ib), np.left_shift(ia, ib))
    assert np.array_equal(_np_right_shift(np, ia, ib), np.right_shift(ia, ib))

    assert np.array_equal(_np_logaddexp(np, a, b), np.logaddexp(a, b))
    assert np.array_equal(_np_logaddexp2(np, a, b), np.logaddexp2(a, b))

    assert np.array_equal(_np_nan_to_num(np, np.array([np.nan, 1.0])), np.nan_to_num([np.nan, 1.0]))

    f1, f2 = _np_frexp(np, a)
    expected_f1, expected_f2 = np.frexp(a)
    assert np.array_equal(f1, expected_f1)
    assert np.array_equal(f2, expected_f2)

    assert np.array_equal(_np_clip(np, a, 2.0, 3.0), np.clip(a, 2.0, 3.0))
    assert np.array_equal(_np_amax(np, a), np.amax(a))
    assert np.array_equal(_np_amin(np, a), np.amin(a))

    x = np.array([0.1, 0.9])
    assert np.allclose(_np_logit(np, x), np.log(x / (1.0 - x)))

    # stubs return zeros
    assert np.array_equal(_np_polygamma(np, a), np.zeros_like(a))
    assert np.array_equal(_np_zeta(np, a), np.zeros_like(a))

    assert np.array_equal(_eager_remainder(np, a, b), np.remainder(a, b))
