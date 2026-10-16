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
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def test_math_binary_ops() -> None:
    """Test math binary ops."""
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])
    ia = np.array([2, 4])
    ib = np.array([1, 2])

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_add)(np, a, b), a + b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_subtract)(np, a, b), a - b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_multiply)(np, a, b), a * b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_true_divide)(np, a, b), a / b)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_maximum)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.maximum)(a, b))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_minimum)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.minimum)(a, b))

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_bitwise_and)(np, ia, ib), cast("Callable[..., EagerTensorProtocol]", np.bitwise_and)(ia, ib))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_bitwise_or)(np, ia, ib), cast("Callable[..., EagerTensorProtocol]", np.bitwise_or)(ia, ib))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_bitwise_xor)(np, ia, ib), cast("Callable[..., EagerTensorProtocol]", np.bitwise_xor)(ia, ib))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_left_shift)(np, ia, ib), cast("Callable[..., EagerTensorProtocol]", np.left_shift)(ia, ib))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_right_shift)(np, ia, ib), cast("Callable[..., EagerTensorProtocol]", np.right_shift)(ia, ib))

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_logaddexp)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.logaddexp)(a, b))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_logaddexp2)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.logaddexp2)(a, b))

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_nan_to_num)(np, cast("Callable[..., EagerTensorProtocol]", np.array)([np.nan, 1.0])), cast("Callable[..., EagerTensorProtocol]", np.nan_to_num)([np.nan, 1.0]))

    f1, f2 = cast("Callable[..., EagerTensorProtocol]", _np_frexp)(np, a)
    expected_f1, expected_f2 = np.frexp(a)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(f1, expected_f1)
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(f2, expected_f2)

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_clip)(np, a, 2.0, 3.0), cast("Callable[..., EagerTensorProtocol]", np.clip)(a, 2.0, 3.0))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_amax)(np, a), cast("Callable[..., EagerTensorProtocol]", np.amax)(a))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_amin)(np, a), cast("Callable[..., EagerTensorProtocol]", np.amin)(a))

    x = np.array([0.1, 0.9])
    assert cast("Callable[..., EagerTensorProtocol]", np.allclose)(cast("Callable[..., EagerTensorProtocol]", _np_logit)(np, x), cast("Callable[..., EagerTensorProtocol]", np.log)(x / (1.0 - x)))

    # stubs return zeros
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_polygamma)(np, a), cast("Callable[..., EagerTensorProtocol]", np.zeros_like)(a))
    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _np_zeta)(np, a), cast("Callable[..., EagerTensorProtocol]", np.zeros_like)(a))

    assert cast("Callable[..., EagerTensorProtocol]", np.array_equal)(cast("Callable[..., EagerTensorProtocol]", _eager_remainder)(np, a, b), cast("Callable[..., EagerTensorProtocol]", np.remainder)(a, b))
