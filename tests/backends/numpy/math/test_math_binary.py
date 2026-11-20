"""Behavioral unit tests for math binary."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_binary as bin_module
from ml_switcheroo_compiler.backends.eager.reference_math.math_binary import (
    _eager_remainder,
    _np_add,
    _np_amax,
    _np_amin,
    _np_bitwise_and,
    _np_bitwise_or,
    _np_bitwise_xor,
    _np_frexp,
    _np_left_shift,
    _np_logaddexp,
    _np_logaddexp2,
    _np_logit,
    _np_maximum,
    _np_minimum,
    _np_multiply,
    _np_nan_to_num,
    _np_right_shift,
    _np_subtract,
    _np_true_divide,
)
from ml_switcheroo_compiler.backends.eager.reference_math.math_general import _np_clip
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.numpy.eager.scipy_special_kernels import _np_polygamma, _np_zeta
from tests.testing_utils import coerce_operand_list, require_callable, unpack_pair

if typing.TYPE_CHECKING:
    pass


class _TestMathBinaryOpsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_binary_ops` split helpers."""

    a: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    b: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    i_a: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    prob: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]


def test_math_binary_ops() -> None:
    """Test math binary ops."""
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])
    ia = np.array([2, 4])
    ib = np.array([1, 2])

    assert np.array_equal(ensure_ndarray(_np_add(np, a, b)), a + b)
    assert np.array_equal(ensure_ndarray(_np_subtract(np, a, b)), a - b)
    assert np.array_equal(ensure_ndarray(_np_multiply(np, a, b)), a * b)
    assert np.array_equal(ensure_ndarray(_np_true_divide(np, a, b)), a / b)
    assert np.array_equal(ensure_ndarray(_np_maximum(np, a, b)), np.maximum(a, b))
    assert np.array_equal(ensure_ndarray(_np_minimum(np, a, b)), np.minimum(a, b))

    assert np.array_equal(ensure_ndarray(_np_bitwise_and(np, ia, ib)), np.bitwise_and(ia, ib))
    assert np.array_equal(ensure_ndarray(_np_bitwise_or(np, ia, ib)), np.bitwise_or(ia, ib))
    assert np.array_equal(ensure_ndarray(_np_bitwise_xor(np, ia, ib)), np.bitwise_xor(ia, ib))
    assert np.array_equal(ensure_ndarray(_np_left_shift(np, ia, ib)), np.left_shift(ia, ib))
    assert np.array_equal(ensure_ndarray(_np_right_shift(np, ia, ib)), np.right_shift(ia, ib))

    assert np.array_equal(ensure_ndarray(_np_logaddexp(np, a, b)), np.logaddexp(a, b))
    assert np.array_equal(ensure_ndarray(_np_logaddexp2(np, a, b)), np.logaddexp2(a, b))

    assert np.array_equal(
        ensure_ndarray(_np_nan_to_num(np, np.array([np.nan, 1.0]))),
        np.nan_to_num([np.nan, 1.0]),
    )

    f1, f2 = unpack_pair(_np_frexp(np, a))
    expected_f1, expected_f2 = np.frexp(a)
    assert np.array_equal(ensure_ndarray(f1), expected_f1)
    assert np.array_equal(ensure_ndarray(f2), expected_f2)

    assert np.array_equal(ensure_ndarray(_np_clip(np, a, 2.0, 3.0)), np.clip(a, 2.0, 3.0))
    assert np.array_equal(ensure_ndarray(_np_amax(np, a)), np.amax(a))
    assert np.array_equal(ensure_ndarray(_np_amin(np, a)), np.amin(a))

    x = np.array([0.1, 0.9])
    assert np.allclose(ensure_ndarray(_np_logit(np, x)), np.log(x / (1.0 - x)))

    # stubs return zeros
    assert np.array_equal(_np_polygamma(np, a), np.zeros_like(a))
    assert np.array_equal(_np_zeta(np, a), np.zeros_like(a))

    assert np.array_equal(ensure_ndarray(_eager_remainder(np, a, b)), np.remainder(a, b))


def _test_math_binary_ops_part_1(_state: _TestMathBinaryOpsState) -> None:
    """Execute part 1 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])

    assert np.allclose(ensure_ndarray(bin_module._np_add(np, a, b)), [3.0, 6.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Add")(np, a, b), [3.0, 6.0])

    assert np.allclose(ensure_ndarray(bin_module._np_subtract(np, a, b)), [1.0, 2.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Subtract")(np, a, b), [1.0, 2.0])

    assert np.allclose(ensure_ndarray(bin_module._np_multiply(np, a, b)), [2.0, 8.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Multiply")(np, a, b), [2.0, 8.0])

    assert np.allclose(ensure_ndarray(bin_module._np_true_divide(np, a, b)), [2.0, 2.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("TrueDivide")(np, a, b), [2.0, 2.0])

    assert np.allclose(ensure_ndarray(bin_module._np_maximum(np, a, b)), [2.0, 4.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Maximum")(np, a, b), [2.0, 4.0])

    assert np.allclose(ensure_ndarray(bin_module._np_minimum(np, a, b)), [1.0, 2.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Minimum")(np, a, b), [1.0, 2.0])

    i_a = np.array([1, 3], dtype=np.int32)
    i_b = np.array([2, 3], dtype=np.int32)

    assert np.array_equal(ensure_ndarray(bin_module._np_bitwise_and(np, i_a, i_b)), [0, 3])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("BitwiseAnd")(np, i_a, i_b), [0, 3])

    assert np.array_equal(ensure_ndarray(bin_module._np_bitwise_or(np, i_a, i_b)), [3, 3])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("BitwiseOr")(np, i_a, i_b), [3, 3])

    assert np.array_equal(ensure_ndarray(bin_module._np_bitwise_xor(np, i_a, i_b)), [3, 0])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("BitwiseXor")(np, i_a, i_b), [3, 0])
    _state.update({"a": a, "b": b, "i_a": i_a})


def _test_math_binary_ops_part_2(_state: _TestMathBinaryOpsState) -> None:
    """Execute part 2 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    a, b, i_a = _state["a"], _state["b"], _state["i_a"]
    assert np.array_equal(ensure_ndarray(bin_module._np_left_shift(np, i_a, 1)), [2, 6])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("LeftShift")(np, i_a, 1), [2, 6])

    assert np.array_equal(ensure_ndarray(bin_module._np_right_shift(np, i_a, 1)), [0, 1])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("RightShift")(np, i_a, 1), [0, 1])

    assert np.allclose(ensure_ndarray(bin_module._np_logaddexp(np, a, b)), np.logaddexp(a, b))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Logaddexp")(np, a, b), np.logaddexp(a, b))

    assert np.allclose(ensure_ndarray(bin_module._np_logaddexp2(np, a, b)), np.logaddexp2(a, b))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Logaddexp2")(np, a, b), np.logaddexp2(a, b))

    nan_arr = np.array([np.nan, 1.0])
    assert np.allclose(ensure_ndarray(bin_module._np_nan_to_num(np, nan_arr)), [0.0, 1.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("NanToNum")(np, nan_arr), [0.0, 1.0])

    mant, exp = unpack_pair(bin_module._np_frexp(np, a))
    assert len(coerce_operand_list(mant)) == 2
    r_mant, _ = require_callable(numpy_eager_registry.get)("Frexp")(np, a)
    assert len(r_mant) == 2

    assert np.allclose(ensure_ndarray(bin_module._np_clip(np, a, 1.5, 3.0)), [2.0, 3.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Clip")(np, a, 1.5, 3.0), [2.0, 3.0])

    assert bin_module._np_amax(np, a) == 4.0
    assert require_callable(numpy_eager_registry.get)("Amax")(np, a) == 4.0

    assert bin_module._np_amin(np, a) == 2.0
    assert require_callable(numpy_eager_registry.get)("Amin")(np, a) == 2.0

    prob = np.array([0.5, 0.8])
    _state.update({"prob": prob})
    _state.update({"a": a, "b": b, "i_a": i_a})


def _test_math_binary_ops_part_3(_state: _TestMathBinaryOpsState) -> None:
    """Execute part 3 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    prob = _state["prob"]
    assert np.allclose(ensure_ndarray(bin_module._np_logit(np, prob)), np.log(prob / (1.0 - prob)))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Logit")(np, prob), np.log(prob / (1.0 - prob)))

    # Polygamma branches
    from ml_switcheroo_compiler.backends.numpy.eager import scipy_special_kernels as sp_mod

    assert sp_mod._np_polygamma(np, 0, 1.0) is not None
    assert sp_mod._np_polygamma(np, 0, x=1.0) is not None
    assert np.array_equal(sp_mod._np_polygamma(np, np.array([0])), [0])
    assert require_callable(numpy_eager_registry.get)("Polygamma")(np, 0, 1.0) is not None
    assert sp_mod._np_zeta(np, 2.0, 1.0) is not None
    assert sp_mod._np_zeta(np, 2.0, q=1.0) is not None
    assert np.array_equal(sp_mod._np_zeta(np, np.array([2.0])), [0.0])
    assert require_callable(numpy_eager_registry.get)("Zeta")(np, 2.0, 1.0) is not None

    # Remainder
    assert np.allclose(ensure_ndarray(bin_module._eager_remainder(np, 7, 3)), 1)
    assert np.allclose(require_callable(numpy_eager_registry.get)("Remainder")(np, 7, 3), 1)


def test_math_binary_ops_ext() -> None:
    """Test binary operations directly and through eager registry.

    Returns:
        None
    """
    _state: _TestMathBinaryOpsState = {}
    _test_math_binary_ops_part_1(_state)
    _test_math_binary_ops_part_2(_state)
    _test_math_binary_ops_part_3(_state)
    assert len(_state) > 0
