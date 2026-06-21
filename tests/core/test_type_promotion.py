"""Unit tests for the data type promotion functionality.

This module contains test cases to verify that different data types (DTypes) are
promoted correctly according to the type promotion rules of the library, and that
invalid promotions raise the appropriate errors.
"""

import pytest

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import DTypePromotionError
from ml_switcheroo_compiler.core.type_promotion import promote_types


def test_promote_types() -> None:
    """Tests the type promotion logic for various combinations of DType values.

    Verifies that identical types, mixed precision types, and mixed kind types
    promote correctly according to the defined promotion rules. Also ensures
    that promoting an invalid or unknown type raises a DTypePromotionError

    Returns:
    None
    """
    # Test with jax_enable_x64 = False (default)
    config.jax_enable_x64 = False

    assert promote_types(DType.Float32, DType.Float32) == DType.Float32
    # NumPy: Int32 + Float32 -> Float64 -> (clamped) Float32
    assert promote_types(DType.Int32, DType.Float32) == DType.Float32
    # NumPy: Float16 + Float32 -> Float32
    assert promote_types(DType.Float16, DType.Float32) == DType.Float32
    assert promote_types(DType.BFloat16, DType.Float32) == DType.Float32
    # NumPy: Float16 + BFloat16 -> Float32 (approx)
    assert promote_types(DType.Float16, DType.BFloat16) == DType.BFloat16
    assert promote_types(DType.Int16, DType.Int32) == DType.Int32
    # NumPy: Float32 + Complex64 -> Complex64
    assert promote_types(DType.Float32, DType.Complex64) == DType.Complex64
    # NumPy: Float64 + Complex64 -> Complex128 -> (clamped) Complex64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex64
    assert promote_types(DType.Float64, DType.Float32) == DType.Float32

    # Test with jax_enable_x64 = True
    config.jax_enable_x64 = True

    # NumPy: Int32 + Float32 -> Float64
    assert promote_types(DType.Int32, DType.Float32) == DType.Float64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex128
    assert promote_types(DType.Float64, DType.Float32) == DType.Float64

    with pytest.raises(DTypePromotionError):
        promote_types("unknown", DType.Float32)

    config.jax_enable_x64 = False
