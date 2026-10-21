"""Tests for test_math_unary_coverage."""

from __future__ import annotations

import types
import typing

import numpy as np

from ml_switcheroo_compiler.backends.eager_registry import numpy_eager_registry
from ml_switcheroo_compiler.backends.numpy.eager import math_unary as unary_module
from tests.testing_utils import as_callable, as_mod

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_math_unary_ops_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 1 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    x = np.array([1.0, 2.0])

    assert as_callable(np.allclose)(as_callable(unary_module._np_exp)(np, x), as_callable(np.exp)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Exp")(np, x), as_callable(np.exp)(x))

    assert as_callable(np.allclose)(as_callable(unary_module._np_log)(np, x), as_callable(np.log)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Log")(np, x), as_callable(np.log)(x))

    assert as_callable(np.allclose)(as_callable(unary_module._np_log1p)(np, x), as_callable(np.log1p)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Log1p")(np, x), as_callable(np.log1p)(x))

    assert as_callable(np.allclose)(as_callable(unary_module._np_round)(np, x), as_callable(np.round)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Round")(np, x), as_callable(np.round)(x))

    # Erf branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erf(np, 0.5), float)
    assert as_callable(len)(as_callable(unary_module._np_erf)(np, x)) == 2
    assert isinstance(as_callable(numpy_eager_registry.get)("Erf")(np, 0.5), float)
    no_asarray_mod = types.SimpleNamespace()
    assert isinstance(unary_module._np_erf(as_mod(no_asarray_mod), 0.5), float)

    # Erfc branches: scalar vs array and without asarray
    assert isinstance(unary_module._np_erfc(np, 0.5), float)
    assert as_callable(len)(as_callable(unary_module._np_erfc)(np, x)) == 2
    assert isinstance(as_callable(numpy_eager_registry.get)("Erfc")(np, 0.5), float)
    assert isinstance(unary_module._np_erfc(as_mod(no_asarray_mod), 0.5), float)

    # Erfinv
    assert as_callable(np.allclose)(as_callable(unary_module._np_erfinv)(np, as_callable(np.array)([0.0])), [0.0])
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Erfinv")(np, as_callable(np.array)([0.0])), [0.0])

    # Igamma branches
    assert unary_module._np_igamma(np, 1.0, 2.0) is not None
    assert unary_module._np_igamma(np, 1.0, x=2.0) is not None
    getattr(_state, "update")({"no_asarray_mod": no_asarray_mod, "x": x})
    _state.update(locals())


def _test_math_unary_ops_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    x = _state["x"]
    assert as_callable(numpy_eager_registry.get)("Igamma")(np, 1.0, 2.0) is not None

    # Igammac branches
    assert unary_module._np_igammac(np, 1.0, 2.0) is not None
    assert unary_module._np_igammac(np, 1.0, x=2.0) is not None
    assert as_callable(numpy_eager_registry.get)("Igammac")(np, 1.0, 2.0) is not None

    # BitwiseNot
    i_x = np.array([1, 2], dtype=np.int32)
    assert as_callable(np.array_equal)(as_callable(unary_module._np_bitwise_not)(np, i_x), ~i_x)
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("BitwiseNot")(np, i_x), ~i_x)

    # Angle
    c_x = np.array([1 + 1j, 1 - 1j])
    assert as_callable(np.allclose)(as_callable(unary_module._np_angle)(np, c_x), as_callable(np.angle)(c_x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Angle")(np, c_x), as_callable(np.angle)(c_x))

    # Expm1
    assert as_callable(np.allclose)(as_callable(unary_module._np_expm1)(np, x), as_callable(np.expm1)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Expm1")(np, x), as_callable(np.expm1)(x))

    # Log10, Log2, Exp2
    assert as_callable(np.allclose)(as_callable(unary_module._np_log10)(np, x), as_callable(np.log10)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Log10")(np, x), as_callable(np.log10)(x))

    assert as_callable(np.allclose)(as_callable(unary_module._np_log2)(np, x), as_callable(np.log2)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Log2")(np, x), as_callable(np.log2)(x))

    assert as_callable(np.allclose)(as_callable(unary_module._np_exp2)(np, x), as_callable(np.exp2)(x))
    assert as_callable(np.allclose)(as_callable(numpy_eager_registry.get)("Exp2")(np, x), as_callable(np.exp2)(x))

    # Signbit
    s_x = np.array([-1.0, 1.0])
    assert as_callable(np.array_equal)(as_callable(unary_module._np_signbit)(np, s_x), [True, False])
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("Signbit")(np, s_x), [True, False])

    # Isnan, Isinf, Isfinite
    mix = np.array([np.nan, np.inf, 1.0])
    getattr(_state, "update")({"c_x": c_x, "i_x": i_x, "mix": mix, "s_x": s_x})
    _state.update(locals())


def _test_math_unary_ops_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_math_unary_ops.

    Args:
        _state: Shared test state dictionary.
    """
    mix = _state["mix"]
    assert as_callable(np.array_equal)(as_callable(unary_module._np_isnan)(np, mix), [True, False, False])
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("Isnan")(np, mix), [True, False, False])

    assert as_callable(np.array_equal)(as_callable(unary_module._np_isinf)(np, mix), [False, True, False])
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("Isinf")(np, mix), [False, True, False])

    assert as_callable(np.array_equal)(as_callable(unary_module._np_isfinite)(np, mix), [False, False, True])
    assert as_callable(np.array_equal)(as_callable(numpy_eager_registry.get)("Isfinite")(np, mix), [False, False, True])


def test_math_unary_ops() -> None:
    """Test unary operations directly and through eager registry.

    Returns:
        None
    """
    _state: dict[str, EagerTensorProtocol] = {}
    _test_math_unary_ops_part_1(_state)
    _test_math_unary_ops_part_2(_state)
    _test_math_unary_ops_part_3(_state)
    assert len(_state) > 0
