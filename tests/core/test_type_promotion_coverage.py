"""Provides required module functionality."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.type_promotion import promote_types


def test_type_promotion_coverage_brute() -> None:
    """Execute the requested function."""
    # Let's test the `if rank1 > rank2` in Integer promotion
    assert promote_types(DType.Int32, DType.Int8) == DType.Int32
    assert promote_types(DType.Int8, DType.Int32) == DType.Int32

    # We will use mock patch for `is_int1 or is_int2`? We can't mock local variables.


def test_bool_promotion() -> None:
    """Test boolean promotion."""
    assert promote_types("bool", "bool") == "bool"
