"""Tests for numpy eager math binary ops."""

from __future__ import annotations

import typing

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
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    pass


def test_math_binary_ops() -> None:
    """Test math binary ops."""
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])
    ia = np.array([2, 4])
    ib = np.array([1, 2])

    assert as_callable(np.array_equal)(as_callable(_np_add)(np, a, b), a + b)
    assert as_callable(np.array_equal)(as_callable(_np_subtract)(np, a, b), a - b)
    assert as_callable(np.array_equal)(as_callable(_np_multiply)(np, a, b), a * b)
    assert as_callable(np.array_equal)(as_callable(_np_true_divide)(np, a, b), a / b)
    assert as_callable(np.array_equal)(as_callable(_np_maximum)(np, a, b), as_callable(np.maximum)(a, b))
    assert as_callable(np.array_equal)(as_callable(_np_minimum)(np, a, b), as_callable(np.minimum)(a, b))

    assert as_callable(np.array_equal)(as_callable(_np_bitwise_and)(np, ia, ib), as_callable(np.bitwise_and)(ia, ib))
    assert as_callable(np.array_equal)(as_callable(_np_bitwise_or)(np, ia, ib), as_callable(np.bitwise_or)(ia, ib))
    assert as_callable(np.array_equal)(as_callable(_np_bitwise_xor)(np, ia, ib), as_callable(np.bitwise_xor)(ia, ib))
    assert as_callable(np.array_equal)(as_callable(_np_left_shift)(np, ia, ib), as_callable(np.left_shift)(ia, ib))
    assert as_callable(np.array_equal)(as_callable(_np_right_shift)(np, ia, ib), as_callable(np.right_shift)(ia, ib))

    assert as_callable(np.array_equal)(as_callable(_np_logaddexp)(np, a, b), as_callable(np.logaddexp)(a, b))
    assert as_callable(np.array_equal)(as_callable(_np_logaddexp2)(np, a, b), as_callable(np.logaddexp2)(a, b))

    assert as_callable(np.array_equal)(as_callable(_np_nan_to_num)(np, as_callable(np.array)([np.nan, 1.0])), as_callable(np.nan_to_num)([np.nan, 1.0]))

    f1, f2 = as_callable(_np_frexp)(np, a)
    expected_f1, expected_f2 = np.frexp(a)
    assert as_callable(np.array_equal)(f1, expected_f1)
    assert as_callable(np.array_equal)(f2, expected_f2)

    assert as_callable(np.array_equal)(as_callable(_np_clip)(np, a, 2.0, 3.0), as_callable(np.clip)(a, 2.0, 3.0))
    assert as_callable(np.array_equal)(as_callable(_np_amax)(np, a), as_callable(np.amax)(a))
    assert as_callable(np.array_equal)(as_callable(_np_amin)(np, a), as_callable(np.amin)(a))

    x = np.array([0.1, 0.9])
    assert as_callable(np.allclose)(as_callable(_np_logit)(np, x), as_callable(np.log)(x / (1.0 - x)))

    # stubs return zeros
    assert as_callable(np.array_equal)(as_callable(_np_polygamma)(np, a), as_callable(np.zeros_like)(a))
    assert as_callable(np.array_equal)(as_callable(_np_zeta)(np, a), as_callable(np.zeros_like)(a))

    assert as_callable(np.array_equal)(as_callable(_eager_remainder)(np, a, b), as_callable(np.remainder)(a, b))
