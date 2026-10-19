"""Tests for numpy eager math binary ops."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_math_binary_ops() -> None:
    """Test math binary ops."""
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])
    ia = np.array([2, 4])
    ib = np.array([1, 2])

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_add)(np, a, b), a + b)
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_subtract)(np, a, b), a - b)
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_multiply)(np, a, b), a * b)
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_true_divide)(np, a, b), a / b)
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_maximum)(np, a, b), cast("EagerCallableProtocol", np.maximum)(a, b))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_minimum)(np, a, b), cast("EagerCallableProtocol", np.minimum)(a, b))

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_bitwise_and)(np, ia, ib), cast("EagerCallableProtocol", np.bitwise_and)(ia, ib))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_bitwise_or)(np, ia, ib), cast("EagerCallableProtocol", np.bitwise_or)(ia, ib))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_bitwise_xor)(np, ia, ib), cast("EagerCallableProtocol", np.bitwise_xor)(ia, ib))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_left_shift)(np, ia, ib), cast("EagerCallableProtocol", np.left_shift)(ia, ib))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_right_shift)(np, ia, ib), cast("EagerCallableProtocol", np.right_shift)(ia, ib))

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_logaddexp)(np, a, b), cast("EagerCallableProtocol", np.logaddexp)(a, b))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_logaddexp2)(np, a, b), cast("EagerCallableProtocol", np.logaddexp2)(a, b))

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_nan_to_num)(np, cast("EagerCallableProtocol", np.array)([np.nan, 1.0])), cast("EagerCallableProtocol", np.nan_to_num)([np.nan, 1.0]))

    f1, f2 = cast("EagerCallableProtocol", _np_frexp)(np, a)
    expected_f1, expected_f2 = np.frexp(a)
    assert cast("EagerCallableProtocol", np.array_equal)(f1, expected_f1)
    assert cast("EagerCallableProtocol", np.array_equal)(f2, expected_f2)

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_clip)(np, a, 2.0, 3.0), cast("EagerCallableProtocol", np.clip)(a, 2.0, 3.0))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_amax)(np, a), cast("EagerCallableProtocol", np.amax)(a))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_amin)(np, a), cast("EagerCallableProtocol", np.amin)(a))

    x = np.array([0.1, 0.9])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", _np_logit)(np, x), cast("EagerCallableProtocol", np.log)(x / (1.0 - x)))

    # stubs return zeros
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_polygamma)(np, a), cast("EagerCallableProtocol", np.zeros_like)(a))
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _np_zeta)(np, a), cast("EagerCallableProtocol", np.zeros_like)(a))

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", _eager_remainder)(np, a, b), cast("EagerCallableProtocol", np.remainder)(a, b))
