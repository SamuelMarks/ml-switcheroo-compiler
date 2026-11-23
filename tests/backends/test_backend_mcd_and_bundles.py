"""Unit tests for MinimalCoreEagerPrimitives default derivations, capability bundles, and applicative verification."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from ml_switcheroo_ir import Eq
from ml_switcheroo_ir.typeclasses import Invalid, Valid
from numpy.typing import NDArray

from ml_switcheroo_compiler.backends.eager_registry.types_base import BackendOperand
from ml_switcheroo_compiler.backends.typeclass_verifier import validate_all_typeclasses_checked
from ml_switcheroo_compiler.backends.typeclasses import (
    AOTCompilerDictionary,
    AOTCompilerTypeclass,
    BackendCapabilityBundle,
    CapabilityKind,
    CodeEmitterDictionary,
    CodeEmitterTypeclass,
    CoreEagerTypeclass,
    MinimalCoreEagerPrimitives,
    derive_core_eager_dictionary,
    verify_core_eager_algebraic_laws,
    verify_core_eager_typeclass,
)
from ml_switcheroo_compiler.core.shape import OpValue

FloatArray = NDArray[np.float64]


class _IdentityRunner:
    """Callable object satisfying VariadicCallableProtocol for AOTCompilerDictionary tests."""

    def __call__(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Return first positional argument or 0.

        Args:
            *args (OpValue): Positional operands.
            **kwargs (OpValue): Keyword operands.

        Returns:
            OpValue: First argument or 0.
        """
        _ = kwargs
        return args[0] if args else 0


def _arr64(data: BackendOperand, dtype: str | None) -> FloatArray:
    """Create a float64 numpy array for minimal primitive tests.

    Args:
        data (BackendOperand): Input scalar or sequence.
        dtype (str | None): Optional dtype string.

    Returns:
        FloatArray: Constructed float64 array.
    """
    _ = dtype
    return np.asarray(data, dtype=np.float64)


def test_minimal_core_eager_primitives_derives_all_default_ops() -> None:
    """Verify derive_core_eager_dictionary synthesizes asarray, subtract, sqrt, abs, tanh, and mean."""
    primitives = MinimalCoreEagerPrimitives[FloatArray](
        zeros_fn=lambda shape, _dtype: np.zeros(shape, dtype=np.float64),
        ones_fn=lambda shape, _dtype: np.ones(shape, dtype=np.float64),
        array_fn=_arr64,
        exp_fn=lambda x: np.asarray(np.exp(x), dtype=np.float64),
        log_fn=lambda x: np.asarray(np.log(x), dtype=np.float64),
        sin_fn=lambda x: np.asarray(np.sin(x), dtype=np.float64),
        cos_fn=lambda x: np.asarray(np.cos(x), dtype=np.float64),
        add_fn=lambda a, b: np.asarray(np.add(a, b), dtype=np.float64),
        multiply_fn=lambda a, b: np.asarray(np.multiply(a, b), dtype=np.float64),
        divide_fn=lambda a, b: np.asarray(np.divide(a, b), dtype=np.float64),
        matmul_fn=lambda a, b: np.asarray(np.matmul(a, b), dtype=np.float64),
        reshape_fn=lambda x, shp: np.asarray(np.reshape(x, shp), dtype=np.float64),
        transpose_fn=lambda x, axes: np.asarray(np.transpose(x, axes=axes), dtype=np.float64),
        sum_fn=lambda x, axis, keepdims: np.asarray(np.sum(x, axis=axis, keepdims=keepdims), dtype=np.float64),
    )
    derived = verify_core_eager_typeclass(derive_core_eager_dictionary(primitives))
    assert isinstance(derived, CoreEagerTypeclass)

    arr = derived.asarray([4.0, 9.0], None)
    np.testing.assert_allclose(arr, np.array([4.0, 9.0]))
    np.testing.assert_allclose(derived.sqrt(arr), np.array([2.0, 3.0]))
    np.testing.assert_allclose(derived.abs(np.array([-3.0, 5.0])), np.array([3.0, 5.0]))
    np.testing.assert_allclose(derived.subtract(np.array([10.0, 4.0]), np.array([3.0, 1.5])), np.array([7.0, 2.5]))
    np.testing.assert_allclose(derived.tanh(np.array([0.0, 0.5])), np.tanh(np.array([0.0, 0.5])))

    mat = np.array([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]])
    np.testing.assert_allclose(derived.mean(mat, None, False), np.mean(mat))
    np.testing.assert_allclose(derived.mean(mat, 1, True), np.mean(mat, axis=1, keepdims=True))

    # Also test explicit override branch on all 6 optional methods
    overridden = derive_core_eager_dictionary(
        MinimalCoreEagerPrimitives[FloatArray](
            zeros_fn=primitives.zeros_fn,
            ones_fn=primitives.ones_fn,
            array_fn=primitives.array_fn,
            exp_fn=primitives.exp_fn,
            log_fn=primitives.log_fn,
            sin_fn=primitives.sin_fn,
            cos_fn=primitives.cos_fn,
            add_fn=primitives.add_fn,
            multiply_fn=primitives.multiply_fn,
            divide_fn=primitives.divide_fn,
            matmul_fn=primitives.matmul_fn,
            reshape_fn=primitives.reshape_fn,
            transpose_fn=primitives.transpose_fn,
            sum_fn=primitives.sum_fn,
            asarray_fn=_arr64,
            subtract_fn=lambda a, b: np.asarray(np.subtract(a, b), dtype=np.float64),
            sqrt_fn=lambda x: np.asarray(np.sqrt(x), dtype=np.float64),
            abs_fn=lambda x: np.asarray(np.abs(x), dtype=np.float64),
            tanh_fn=lambda x: np.asarray(np.tanh(x), dtype=np.float64),
            mean_fn=lambda x, axis, keepdims: np.asarray(np.mean(x, axis=axis, keepdims=keepdims), dtype=np.float64),
        )
    )
    assert math.isclose(float(overridden.sqrt(np.array(16.0))), 4.0)


def test_capability_dictionaries_bundle_and_applicative_verifier(tmp_path: Path) -> None:
    """Verify capability dictionaries, BackendCapabilityBundle, and validate_all_typeclasses_checked.

    Args:
        tmp_path (Path): Temporary pytest directory fixture.
    """
    emit_dict = CodeEmitterDictionary(generate_fn=lambda: "def run(x): return x")
    assert isinstance(emit_dict, CodeEmitterTypeclass)
    assert emit_dict.generate() == "def run(x): return x"

    runner = _IdentityRunner()
    aot_dict = AOTCompilerDictionary(compile_aot_fn=lambda: runner)
    assert isinstance(aot_dict, AOTCompilerTypeclass)
    assert aot_dict.compile_aot()(42) == 42

    empty_bundle = BackendCapabilityBundle[FloatArray](backend_name="minimal")
    assert empty_bundle.capabilities() == frozenset()

    primitives = MinimalCoreEagerPrimitives[FloatArray](
        zeros_fn=lambda shape, _dt: np.zeros(shape, dtype=np.float64),
        ones_fn=lambda shape, _dt: np.ones(shape, dtype=np.float64),
        array_fn=_arr64,
        exp_fn=lambda x: np.asarray(np.exp(x), dtype=np.float64),
        log_fn=lambda x: np.asarray(np.log(x), dtype=np.float64),
        sin_fn=lambda x: np.asarray(np.sin(x), dtype=np.float64),
        cos_fn=lambda x: np.asarray(np.cos(x), dtype=np.float64),
        add_fn=lambda a, b: np.asarray(np.add(a, b), dtype=np.float64),
        multiply_fn=lambda a, b: np.asarray(np.multiply(a, b), dtype=np.float64),
        divide_fn=lambda a, b: np.asarray(np.divide(a, b), dtype=np.float64),
        matmul_fn=lambda a, b: np.asarray(np.matmul(a, b), dtype=np.float64),
        reshape_fn=lambda x, shp: np.asarray(np.reshape(x, shp), dtype=np.float64),
        transpose_fn=lambda x, axes: np.asarray(np.transpose(x, axes=axes), dtype=np.float64),
        sum_fn=lambda x, axis, keepdims: np.asarray(np.sum(x, axis=axis, keepdims=keepdims), dtype=np.float64),
    )
    full_bundle = BackendCapabilityBundle[FloatArray](
        backend_name="full_dict_backend",
        code_emitter=emit_dict,
        aot_compiler=aot_dict,
        core_eager=derive_core_eager_dictionary(primitives),
    )
    assert full_bundle.capabilities() == frozenset(
        {
            CapabilityKind.CODE_EMIT,
            CapabilityKind.AOT_COMPILE,
            CapabilityKind.CORE_EAGER,
        }
    )

    valid_res = validate_all_typeclasses_checked(live_backends=("numpy",))
    assert isinstance(valid_res, Valid) and valid_res.value == ("numpy",)

    missing_dir = str(tmp_path / "nonexistent_backends")
    invalid_res = validate_all_typeclasses_checked(backends_dir=missing_dir, live_backends=())
    assert isinstance(invalid_res, Invalid) and len(invalid_res.errors) >= 2

    # Fractional MCD: reciprocal_fn derives divide; omitting both divide_fn & reciprocal_fn raises ValueError
    recip_primitives = MinimalCoreEagerPrimitives[FloatArray](
        zeros_fn=primitives.zeros_fn,
        ones_fn=primitives.ones_fn,
        array_fn=primitives.array_fn,
        exp_fn=primitives.exp_fn,
        log_fn=primitives.log_fn,
        sin_fn=primitives.sin_fn,
        cos_fn=primitives.cos_fn,
        add_fn=primitives.add_fn,
        multiply_fn=primitives.multiply_fn,
        reciprocal_fn=lambda x: np.asarray(1.0 / x, dtype=np.float64),
        matmul_fn=primitives.matmul_fn,
        reshape_fn=primitives.reshape_fn,
        transpose_fn=primitives.transpose_fn,
        sum_fn=primitives.sum_fn,
    )
    recip_dict = derive_core_eager_dictionary(recip_primitives)
    approx_eq: Eq[FloatArray] = Eq(eq=lambda a, b: bool(np.allclose(a, b, atol=1e-9, rtol=1e-9)))
    law_ok = verify_core_eager_algebraic_laws(recip_dict, approx_eq, label="numpy_recip_mcd")
    assert isinstance(law_ok, Valid)

    broken_dict = derive_core_eager_dictionary(
        MinimalCoreEagerPrimitives[FloatArray](
            zeros_fn=primitives.zeros_fn,
            ones_fn=primitives.ones_fn,
            array_fn=primitives.array_fn,
            exp_fn=primitives.exp_fn,
            log_fn=primitives.log_fn,
            sin_fn=primitives.sin_fn,
            cos_fn=primitives.cos_fn,
            add_fn=lambda a, b: np.asarray(a - b, dtype=np.float64),
            multiply_fn=primitives.multiply_fn,
            divide_fn=primitives.divide_fn,
            matmul_fn=primitives.matmul_fn,
            reshape_fn=primitives.reshape_fn,
            transpose_fn=primitives.transpose_fn,
            sum_fn=primitives.sum_fn,
        )
    )
    law_bad = verify_core_eager_algebraic_laws(broken_dict, approx_eq, label="broken_add")
    assert isinstance(law_bad, Invalid) and any("add_commutativity" in e for e in law_bad.errors)

    incomplete = MinimalCoreEagerPrimitives[FloatArray](
        zeros_fn=primitives.zeros_fn,
        ones_fn=primitives.ones_fn,
        array_fn=primitives.array_fn,
        exp_fn=primitives.exp_fn,
        log_fn=primitives.log_fn,
        sin_fn=primitives.sin_fn,
        cos_fn=primitives.cos_fn,
        add_fn=primitives.add_fn,
        multiply_fn=primitives.multiply_fn,
        matmul_fn=primitives.matmul_fn,
        reshape_fn=primitives.reshape_fn,
        transpose_fn=primitives.transpose_fn,
        sum_fn=primitives.sum_fn,
    )
    with pytest.raises(ValueError, match="divide_fn or reciprocal_fn"):
        derive_core_eager_dictionary(incomplete)
