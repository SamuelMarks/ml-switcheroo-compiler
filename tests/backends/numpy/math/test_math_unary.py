"""Behavioral unit tests for math unary."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager.reference_math import math_unary as unary_module
from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import MockTestStub, coerce_operand_list, require_callable


class _TestMathUnaryOpsState(typing.TypedDict, total=False):
    """Typed shared state for `_test_math_unary_ops` split helpers."""

    mix: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]
    x: np.ndarray[
        tuple[int, ...],
        np.dtype[np.float64 | np.float32 | np.int64 | np.int32 | np.uint8 | np.bool_],
    ]


def _test_math_unary_ops_part_1(_state: _TestMathUnaryOpsState) -> None:
    """Execute part 1 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    x = np.array([1.0, 2.0])

    assert np.allclose(ensure_ndarray(unary_module._np_exp(np, x)), np.exp(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Exp")(np, x), np.exp(x))

    assert np.allclose(ensure_ndarray(unary_module._np_log(np, x)), np.log(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Log")(np, x), np.log(x))

    assert np.allclose(ensure_ndarray(unary_module._np_log1p(np, x)), np.log1p(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Log1p")(np, x), np.log1p(x))

    assert np.allclose(ensure_ndarray(unary_module._np_round(np, x)), np.round(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Round")(np, x), np.round(x))

    # Erf branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erf(np, 0.5), float)
    assert len(coerce_operand_list(unary_module._np_erf(np, x))) == 2
    assert isinstance(require_callable(numpy_eager_registry.get)("Erf")(np, 0.5), float)
    no_asarray_mod = MockTestStub()
    assert isinstance(unary_module._np_erf(no_asarray_mod, 0.5), float)

    # Erfc branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erfc(np, 0.5), float)
    assert len(coerce_operand_list(unary_module._np_erfc(np, x))) == 2
    assert isinstance(require_callable(numpy_eager_registry.get)("Erfc")(np, 0.5), float)
    assert isinstance(unary_module._np_erfc(no_asarray_mod, 0.5), float)

    # Erfinv
    from ml_switcheroo_compiler.backends.numpy.eager import scipy_special_kernels as sp_unary

    assert np.allclose(ensure_ndarray(sp_unary._np_erfinv(np, np.array([0.0]))), [0.0])
    assert np.allclose(require_callable(numpy_eager_registry.get)("Erfinv")(np, np.array([0.0])), [0.0])

    # Igamma branches
    assert sp_unary._np_igamma(np, 1.0, 2.0) is not None
    assert sp_unary._np_igamma(np, 1.0, x=2.0) is not None
    _state.update({"x": x})


def _test_math_unary_ops_part_2(_state: _TestMathUnaryOpsState) -> None:
    """Execute part 2 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    from ml_switcheroo_compiler.backends.numpy.eager import scipy_special_kernels as sp_unary

    x = _state["x"]
    assert require_callable(numpy_eager_registry.get)("Igamma")(np, 1.0, 2.0) is not None

    # Igammac branches
    assert sp_unary._np_igammac(np, 1.0, 2.0) is not None
    assert sp_unary._np_igammac(np, 1.0, x=2.0) is not None
    assert require_callable(numpy_eager_registry.get)("Igammac")(np, 1.0, 2.0) is not None

    # BitwiseNot
    i_x = np.array([1, 2], dtype=np.int32)
    assert np.array_equal(ensure_ndarray(unary_module._np_bitwise_not(np, i_x)), ~i_x)
    assert np.array_equal(require_callable(numpy_eager_registry.get)("BitwiseNot")(np, i_x), ~i_x)

    # Angle
    c_x = np.array([1 + 1j, 1 - 1j])
    assert np.allclose(ensure_ndarray(unary_module._np_angle(np, c_x)), np.angle(c_x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Angle")(np, c_x), np.angle(c_x))

    # Expm1
    assert np.allclose(ensure_ndarray(unary_module._np_expm1(np, x)), np.expm1(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Expm1")(np, x), np.expm1(x))

    # Log10, Log2, Exp2
    assert np.allclose(ensure_ndarray(unary_module._np_log10(np, x)), np.log10(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Log10")(np, x), np.log10(x))

    assert np.allclose(ensure_ndarray(unary_module._np_log2(np, x)), np.log2(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Log2")(np, x), np.log2(x))

    assert np.allclose(ensure_ndarray(unary_module._np_exp2(np, x)), np.exp2(x))
    assert np.allclose(require_callable(numpy_eager_registry.get)("Exp2")(np, x), np.exp2(x))

    # Signbit
    s_x = np.array([-1.0, 1.0])
    assert np.array_equal(ensure_ndarray(unary_module._np_signbit(np, s_x)), [True, False])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Signbit")(np, s_x), [True, False])

    # Isnan, Isinf, Isfinite
    mix = np.array([np.nan, np.inf, 1.0])
    _state.update({"mix": mix})
    _state.update({"x": x})


def _test_math_unary_ops_part_3(_state: _TestMathUnaryOpsState) -> None:
    """Execute part 3 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    mix = _state["mix"]
    assert np.array_equal(ensure_ndarray(unary_module._np_isnan(np, mix)), [True, False, False])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Isnan")(np, mix), [True, False, False])

    assert np.array_equal(ensure_ndarray(unary_module._np_isinf(np, mix)), [False, True, False])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Isinf")(np, mix), [False, True, False])

    assert np.array_equal(ensure_ndarray(unary_module._np_isfinite(np, mix)), [False, False, True])
    assert np.array_equal(require_callable(numpy_eager_registry.get)("Isfinite")(np, mix), [False, False, True])


def test_math_unary_ops() -> None:
    """Test unary operations directly and through eager registry.

    Returns:
        None
    """
    _state: _TestMathUnaryOpsState = {}
    _test_math_unary_ops_part_1(_state)
    _test_math_unary_ops_part_2(_state)
    _test_math_unary_ops_part_3(_state)
    assert len(_state) > 0
