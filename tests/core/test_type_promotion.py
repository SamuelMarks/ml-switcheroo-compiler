"""Unit tests for the data type promotion functionality.

This module contains test cases to verify that different data types (DTypes) are
promoted correctly according to the type promotion rules of the library, and that
invalid promotions raise the appropriate errors.
"""

import pytest

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
    assert promote_types(DType.Float32, DType.Float32) == DType.Float32
    assert promote_types(DType.Int32, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.Float32) == DType.Float32
    assert promote_types(DType.BFloat16, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.BFloat16) == DType.Float32
    assert promote_types(DType.Int16, DType.Int32) == DType.Int32
    assert promote_types(DType.Float32, DType.Complex64) == DType.Complex64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex128
    assert promote_types(DType.Bool, DType.Int32) == DType.Int32

    # Float promotions missing coverage
    assert promote_types(DType.Float64, DType.Float32) == DType.Float64
    assert promote_types(DType.BFloat16, DType.Float16) == DType.Float32
    # float vs int fallback (if rank1 > rank2)
    assert promote_types(DType.Float32, DType.Int16) == DType.Float32
    assert promote_types(DType.Int16, DType.Float32) == DType.Float32

    # Complex promotions
    assert promote_types(DType.Complex128, DType.Complex64) == DType.Complex128

    # Integer promotions
    assert promote_types(DType.Int64, DType.Int32) == DType.Int64
    assert promote_types(DType.Int32, DType.Int64) == DType.Int64

    # Different families generic fallback
    assert promote_types(DType.Complex64, DType.Int8) == DType.Complex64
    assert promote_types(DType.Int8, DType.Complex64) == DType.Complex64

    # Float vs Int fallback missing coverage (lines 80-82)
    # Neither Float64 nor Float32 nor Float16/BFloat16 mixed
    assert promote_types(DType.Float16, DType.Int32) == DType.Float16
    assert promote_types(DType.Int32, DType.Float16) == DType.Float16
    assert promote_types(DType.BFloat16, DType.Int64) == DType.BFloat16

    # Generic fallback missing coverage (lines 91-93)
    # When neither are complex, float, or int. Bool is the only one left.
    assert promote_types(DType.Bool, DType.Bool) == DType.Bool

    with pytest.raises(DTypePromotionError):
        promote_types("unknown", DType.Float32)
