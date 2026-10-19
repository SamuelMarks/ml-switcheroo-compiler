"""Tests for test_math_binary_coverage."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_binary as bin_module

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol


def _test_math_binary_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    a = np.array([2.0, 4.0])
    b = np.array([1.0, 2.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_add)(np, a, b), [3.0, 6.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Add")(np, a, b), [3.0, 6.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_subtract)(np, a, b), [1.0, 2.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Subtract")(np, a, b), [1.0, 2.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_multiply)(np, a, b), [2.0, 8.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Multiply")(np, a, b), [2.0, 8.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_true_divide)(np, a, b), [2.0, 2.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("TrueDivide")(np, a, b), [2.0, 2.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_maximum)(np, a, b), [2.0, 4.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Maximum")(np, a, b), [2.0, 4.0])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_minimum)(np, a, b), [1.0, 2.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Minimum")(np, a, b), [1.0, 2.0])

    i_a = np.array([1, 3], dtype=np.int32)
    i_b = np.array([2, 3], dtype=np.int32)

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_bitwise_and)(np, i_a, i_b), [0, 3])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", numpy_eager_registry.get)("BitwiseAnd")(np, i_a, i_b), [0, 3])

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_bitwise_or)(np, i_a, i_b), [3, 3])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", numpy_eager_registry.get)("BitwiseOr")(np, i_a, i_b), [3, 3])

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_bitwise_xor)(np, i_a, i_b), [3, 0])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", numpy_eager_registry.get)("BitwiseXor")(np, i_a, i_b), [3, 0])
    getattr(_state, "update")({"a": a, "b": b, "i_a": i_a, "i_b": i_b})
    _state.update(locals())


def _test_math_binary_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    a, b, i_a = _state["a"], _state["b"], _state["i_a"]
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_left_shift)(np, i_a, 1), [2, 6])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", numpy_eager_registry.get)("LeftShift")(np, i_a, 1), [2, 6])

    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_right_shift)(np, i_a, 1), [0, 1])
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", numpy_eager_registry.get)("RightShift")(np, i_a, 1), [0, 1])

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_logaddexp)(np, a, b), cast("EagerCallableProtocol", np.logaddexp)(a, b))
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Logaddexp")(np, a, b), cast("EagerCallableProtocol", np.logaddexp)(a, b))

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_logaddexp2)(np, a, b), cast("EagerCallableProtocol", np.logaddexp2)(a, b))
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Logaddexp2")(np, a, b), cast("EagerCallableProtocol", np.logaddexp2)(a, b))

    nan_arr = np.array([np.nan, 1.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_nan_to_num)(np, nan_arr), [0.0, 1.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("NanToNum")(np, nan_arr), [0.0, 1.0])

    mant, exp = cast("EagerCallableProtocol", bin_module._np_frexp)(np, a)
    assert cast("EagerCallableProtocol", len)(mant) == 2
    r_mant, _ = cast("EagerCallableProtocol", numpy_eager_registry.get)("Frexp")(np, a)
    assert cast("EagerCallableProtocol", len)(r_mant) == 2

    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_clip)(np, a, 1.5, 3.0), [2.0, 3.0])
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Clip")(np, a, 1.5, 3.0), [2.0, 3.0])

    assert bin_module._np_amax(np, a) == 4.0
    assert cast("EagerCallableProtocol", numpy_eager_registry.get)("Amax")(np, a) == 4.0

    assert bin_module._np_amin(np, a) == 2.0
    assert cast("EagerCallableProtocol", numpy_eager_registry.get)("Amin")(np, a) == 2.0

    prob = np.array([0.5, 0.8])
    getattr(_state, "update")({"_": _, "exp": exp, "mant": mant, "nan_arr": nan_arr})
    getattr(_state, "update")({"prob": prob, "r_mant": r_mant})
    _state.update(locals())


def _test_math_binary_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_binary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    prob = _state["prob"]
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._np_logit)(np, prob), cast("EagerCallableProtocol", np.log)(prob / (1.0 - prob)))
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Logit")(np, prob), cast("EagerCallableProtocol", np.log)(prob / (1.0 - prob)))

    # Polygamma branches
    assert bin_module._np_polygamma(np, 0, 1.0) is not None
    assert bin_module._np_polygamma(np, 0, x=1.0) is not None
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_polygamma)(np, cast("EagerCallableProtocol", np.array)([0])), [0])
    assert cast("EagerCallableProtocol", numpy_eager_registry.get)("Polygamma")(np, 0, 1.0) is not None
    assert bin_module._np_zeta(np, 2.0, 1.0) is not None
    assert bin_module._np_zeta(np, 2.0, q=1.0) is not None
    assert cast("EagerCallableProtocol", np.array_equal)(cast("EagerCallableProtocol", bin_module._np_zeta)(np, cast("EagerCallableProtocol", np.array)([2.0])), [0.0])
    assert cast("EagerCallableProtocol", numpy_eager_registry.get)("Zeta")(np, 2.0, 1.0) is not None

    # Remainder
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", bin_module._eager_remainder)(np, 7, 3), 1)
    assert cast("EagerCallableProtocol", np.allclose)(cast("EagerCallableProtocol", numpy_eager_registry.get)("Remainder")(np, 7, 3), 1)


def test_math_binary_ops() -> None:
    """Test binary operations directly and through eager registry.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_binary_ops_part_1(_state)
    _test_math_binary_ops_part_2(_state)
    _test_math_binary_ops_part_3(_state)
    assert len(_state) > 0
