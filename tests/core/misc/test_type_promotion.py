"""Unit tests for data type promotion rules and lattice loading."""

from __future__ import annotations

import typing
from unittest.mock import mock_open, patch

import pytest

import ml_switcheroo_compiler.core.type_promotion as tp
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import DTypePromotionError
from ml_switcheroo_compiler.core.type_promotion import promote_types

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol


def test_type_promotion_complex128_downgrade() -> None:
    """Test Complex128 downgrade to Complex64 when jax_enable_x64 is False."""
    with ConfigContext(jax_enable_x64=False):
        res = promote_types(DType.Complex128, DType.Complex128)
        assert res == DType.Complex64


def test_promote_types() -> None:
    """Test type promotion across DType combinations and x64 configurations."""
    config.jax_enable_x64 = False
    assert promote_types(DType.Float32, DType.Float32) == DType.Float32
    assert promote_types(DType.Int32, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.Float32) == DType.Float32
    assert promote_types(DType.BFloat16, DType.Float32) == DType.Float32
    assert promote_types(DType.Float16, DType.BFloat16) == DType.BFloat16
    assert promote_types(DType.Int16, DType.Int32) == DType.Int32
    assert promote_types(DType.Float32, DType.Complex64) == DType.Complex64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex64
    assert promote_types(DType.Float64, DType.Float32) == DType.Float32
    config.jax_enable_x64 = True
    assert promote_types(DType.Int32, DType.Float32) == DType.Float64
    assert promote_types(DType.Float64, DType.Complex64) == DType.Complex128
    assert promote_types(DType.Float64, DType.Float32) == DType.Float64
    with pytest.raises(DTypePromotionError):
        typing.cast("EagerCallableProtocol", promote_types)("unknown", DType.Float32)
    config.jax_enable_x64 = False


def test_type_promotion_coverage_brute() -> None:
    """Test symmetric integer promotion."""
    assert promote_types(DType.Int32, DType.Int8) == DType.Int32
    assert promote_types(DType.Int8, DType.Int32) == DType.Int32


def test_bool_promotion() -> None:
    """Test boolean promotion."""
    assert promote_types(DType.Bool, DType.Bool) == DType.Bool
    with pytest.raises(DTypePromotionError):
        typing.cast("EagerCallableProtocol", promote_types)("bool", "bool")


def test_promote_complex128_downcast() -> None:
    """Test Complex128 + Complex64 promotion under default x64=False."""
    with ConfigContext(jax_enable_x64=False):
        assert promote_types(DType.Complex128, DType.Complex64) == DType.Complex64


def test_type_promotion_129() -> None:
    """Test type promotion edge case with custom equality object."""

    class FakeType:
        """Custom object equal to DType.Bool but not equal to another FakeType."""

        def __eq__(self, other: object) -> bool:
            """Test function __eq__.

            Args:
                other: Test parameter `other`.
            """
            if type(other) is type(self):
                return False
            return other == DType.Bool

        def __hash__(self) -> int:
            """Test function __hash__."""
            return hash(DType.Bool)

    res = typing.cast("EagerCallableProtocol", tp.promote_types)(typing.cast("EagerCallableProtocol", FakeType)(), typing.cast("EagerCallableProtocol", FakeType)())
    assert isinstance(res, FakeType)


def test_declarative_yaml_type_promotion_consistency() -> None:
    """Validate declarative type promotion rules across all internal backend types."""
    with ConfigContext(jax_enable_x64=True):
        assert promote_types(DType.Bool, DType.Bool) == DType.Bool
        assert promote_types(DType.Bool, DType.Int32) == DType.Int32
        assert promote_types(DType.Bool, DType.Float16) == DType.Float16
        assert promote_types(DType.Bool, DType.BFloat16) == DType.BFloat16
        assert promote_types(DType.Bool, DType.Float32) == DType.Float32
        assert promote_types(DType.Bool, DType.Float64) == DType.Float64

        assert promote_types(DType.Int8, DType.Int16) == DType.Int16
        assert promote_types(DType.Int16, DType.Int32) == DType.Int32
        assert promote_types(DType.Int32, DType.Int64) == DType.Int64

        assert promote_types(DType.Float16, DType.Float32) == DType.Float32
        assert promote_types(DType.BFloat16, DType.Float32) == DType.Float32
        assert promote_types(DType.Float32, DType.Float64) == DType.Float64
        assert promote_types(DType.Float16, DType.Float64) == DType.Float64

        assert promote_types(DType.Int16, DType.Float32) == DType.Float32
        assert promote_types(DType.Int32, DType.Float32) == DType.Float64
        assert promote_types(DType.Int64, DType.Float32) == DType.Float64


def test_load_declarative_promotion_lattice_branches() -> None:
    """Test branches of _load_declarative_promotion_lattice including non-existent file and invalid DType values."""
    with patch("os.path.exists", return_value=False):
        tp._load_declarative_promotion_lattice()
        assert isinstance(tp._PROMOTION_LATTICE, dict)

    fake_yaml = """
lattice:
  invalid_source_dtype_value:
    float32: float32
  float32:
    invalid_target_dtype_value: float32
    int32: invalid_promoted_dtype_value
"""
    with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=fake_yaml)):
        tp._load_declarative_promotion_lattice()
        assert isinstance(tp._PROMOTION_LATTICE, dict)
