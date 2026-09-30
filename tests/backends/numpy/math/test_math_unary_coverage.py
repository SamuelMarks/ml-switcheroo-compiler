"""Tests for test_math_unary_coverage."""

from __future__ import annotations

import types

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_unary as unary_module


def test_math_unary_ops() -> None:
    """Test unary operations directly and through eager registry.

    Returns:
        None
    """
    x = np.array([1.0, 2.0])

    assert np.allclose(unary_module._np_exp(np, x), np.exp(x))
    assert np.allclose(numpy_eager_registry.get("Exp")(np, x), np.exp(x))

    assert np.allclose(unary_module._np_log(np, x), np.log(x))
    assert np.allclose(numpy_eager_registry.get("Log")(np, x), np.log(x))

    assert np.allclose(unary_module._np_log1p(np, x), np.log1p(x))
    assert np.allclose(numpy_eager_registry.get("Log1p")(np, x), np.log1p(x))

    assert np.allclose(unary_module._np_round(np, x), np.round(x))
    assert np.allclose(numpy_eager_registry.get("Round")(np, x), np.round(x))

    # Erf branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erf(np, 0.5), float)
    assert len(unary_module._np_erf(np, x)) == 2
    assert isinstance(numpy_eager_registry.get("Erf")(np, 0.5), float)
    no_asarray_mod = types.SimpleNamespace()
    assert isinstance(unary_module._np_erf(no_asarray_mod, 0.5), float)

    # Erfc branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erfc(np, 0.5), float)
    assert len(unary_module._np_erfc(np, x)) == 2
    assert isinstance(numpy_eager_registry.get("Erfc")(np, 0.5), float)
    assert isinstance(unary_module._np_erfc(no_asarray_mod, 0.5), float)

    # Erfinv
    assert np.allclose(unary_module._np_erfinv(np, np.array([0.0])), [0.0])
    assert np.allclose(numpy_eager_registry.get("Erfinv")(np, np.array([0.0])), [0.0])

    # Igamma branches
    assert unary_module._np_igamma(np, 1.0, 2.0) is not None
    assert unary_module._np_igamma(np, 1.0, x=2.0) is not None
    assert numpy_eager_registry.get("Igamma")(np, 1.0, 2.0) is not None

    # Igammac branches
    assert unary_module._np_igammac(np, 1.0, 2.0) is not None
    assert unary_module._np_igammac(np, 1.0, x=2.0) is not None
    assert numpy_eager_registry.get("Igammac")(np, 1.0, 2.0) is not None

    # BitwiseNot
    i_x = np.array([1, 2], dtype=np.int32)
    assert np.array_equal(unary_module._np_bitwise_not(np, i_x), ~i_x)
    assert np.array_equal(numpy_eager_registry.get("BitwiseNot")(np, i_x), ~i_x)

    # Angle
    c_x = np.array([1 + 1j, 1 - 1j])
    assert np.allclose(unary_module._np_angle(np, c_x), np.angle(c_x))
    assert np.allclose(numpy_eager_registry.get("Angle")(np, c_x), np.angle(c_x))

    # Expm1
    assert np.allclose(unary_module._np_expm1(np, x), np.expm1(x))
    assert np.allclose(numpy_eager_registry.get("Expm1")(np, x), np.expm1(x))

    # Log10, Log2, Exp2
    assert np.allclose(unary_module._np_log10(np, x), np.log10(x))
    assert np.allclose(numpy_eager_registry.get("Log10")(np, x), np.log10(x))

    assert np.allclose(unary_module._np_log2(np, x), np.log2(x))
    assert np.allclose(numpy_eager_registry.get("Log2")(np, x), np.log2(x))

    assert np.allclose(unary_module._np_exp2(np, x), np.exp2(x))
    assert np.allclose(numpy_eager_registry.get("Exp2")(np, x), np.exp2(x))

    # Signbit
    s_x = np.array([-1.0, 1.0])
    assert np.array_equal(unary_module._np_signbit(np, s_x), [True, False])
    assert np.array_equal(numpy_eager_registry.get("Signbit")(np, s_x), [True, False])

    # Isnan, Isinf, Isfinite
    mix = np.array([np.nan, np.inf, 1.0])
    assert np.array_equal(unary_module._np_isnan(np, mix), [True, False, False])
    assert np.array_equal(numpy_eager_registry.get("Isnan")(np, mix), [True, False, False])

    assert np.array_equal(unary_module._np_isinf(np, mix), [False, True, False])
    assert np.array_equal(numpy_eager_registry.get("Isinf")(np, mix), [False, True, False])

    assert np.array_equal(unary_module._np_isfinite(np, mix), [False, False, True])
    assert np.array_equal(numpy_eager_registry.get("Isfinite")(np, mix), [False, False, True])
