# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915

"""Type promotion rules for ml-switcheroo."""

from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import DTypePromotionError

_PROMOTION_TABLE = {
    (DType.Bool, DType.Bool): DType.Bool,
    (DType.Bool, DType.UInt8): DType.UInt8,
    (DType.Bool, DType.Int8): DType.Int8,
    (DType.Bool, DType.Int16): DType.Int16,
    (DType.Bool, DType.Int32): DType.Int32,
    (DType.Bool, DType.Int64): DType.Int64,
    (DType.Bool, DType.Float16): DType.Float16,
    (DType.Bool, DType.Float32): DType.Float32,
    (DType.Bool, DType.Float64): DType.Float64,
    (DType.Bool, DType.Complex64): DType.Complex64,
    (DType.Bool, DType.Complex128): DType.Complex128,
    (DType.Bool, DType.BFloat16): DType.BFloat16,
    (DType.UInt8, DType.UInt8): DType.UInt8,
    (DType.UInt8, DType.Int8): DType.Int16,
    (DType.UInt8, DType.Int16): DType.Int16,
    (DType.UInt8, DType.Int32): DType.Int32,
    (DType.UInt8, DType.Int64): DType.Int64,
    (DType.UInt8, DType.Float16): DType.Float16,
    (DType.UInt8, DType.Float32): DType.Float32,
    (DType.UInt8, DType.Float64): DType.Float64,
    (DType.UInt8, DType.Complex64): DType.Complex64,
    (DType.UInt8, DType.Complex128): DType.Complex128,
    (DType.UInt8, DType.BFloat16): DType.BFloat16,
    (DType.Int4, DType.Int4): DType.Int4,
    (DType.Int4, DType.Int8): DType.Int8,
    (DType.Int4, DType.UInt8): DType.Int16,
    (DType.Int4, DType.Int16): DType.Int16,
    (DType.Int4, DType.Int32): DType.Int32,
    (DType.Int4, DType.Int64): DType.Int64,
    (DType.Int4, DType.Float16): DType.Float16,
    (DType.Int4, DType.Float32): DType.Float32,
    (DType.Int4, DType.Float64): DType.Float64,
    (DType.Int4, DType.BFloat16): DType.BFloat16,
    (DType.Int4, DType.Complex64): DType.Complex64,
    (DType.Int4, DType.Complex128): DType.Complex128,
    (DType.Bool, DType.Int4): DType.Int4,
    (DType.Int8, DType.Int8): DType.Int8,
    (DType.Int8, DType.Int16): DType.Int16,
    (DType.Int8, DType.Int32): DType.Int32,
    (DType.Int8, DType.Int64): DType.Int64,
    (DType.Int8, DType.Float16): DType.Float16,
    (DType.Int8, DType.Float32): DType.Float32,
    (DType.Int8, DType.Float64): DType.Float64,
    (DType.Int8, DType.Complex64): DType.Complex64,
    (DType.Int8, DType.Complex128): DType.Complex128,
    (DType.Int8, DType.BFloat16): DType.BFloat16,
    (DType.Int16, DType.Int16): DType.Int16,
    (DType.Int16, DType.Int32): DType.Int32,
    (DType.Int16, DType.Int64): DType.Int64,
    (DType.Int16, DType.Float16): DType.Float32,
    (DType.Int16, DType.Float32): DType.Float32,
    (DType.Int16, DType.Float64): DType.Float64,
    (DType.Int16, DType.Complex64): DType.Complex64,
    (DType.Int16, DType.Complex128): DType.Complex128,
    (DType.Int16, DType.BFloat16): DType.Float32,
    (DType.Int32, DType.Int32): DType.Int32,
    (DType.Int32, DType.Int64): DType.Int64,
    (DType.Int32, DType.Float16): DType.Float64,
    (DType.Int32, DType.Float32): DType.Float64,
    (DType.Int32, DType.Float64): DType.Float64,
    (DType.Int32, DType.Complex64): DType.Complex128,
    (DType.Int32, DType.Complex128): DType.Complex128,
    (DType.Int32, DType.BFloat16): DType.Float64,
    (DType.Int64, DType.Int64): DType.Int64,
    (DType.Int64, DType.Float16): DType.Float64,
    (DType.Int64, DType.Float32): DType.Float64,
    (DType.Int64, DType.Float64): DType.Float64,
    (DType.Int64, DType.Complex64): DType.Complex128,
    (DType.Int64, DType.Complex128): DType.Complex128,
    (DType.Int64, DType.BFloat16): DType.Float64,
    (DType.Float16, DType.Float16): DType.Float16,
    (DType.Float16, DType.Float32): DType.Float32,
    (DType.Float16, DType.Float64): DType.Float64,
    (DType.Float16, DType.Complex64): DType.Complex64,
    (DType.Float16, DType.Complex128): DType.Complex128,
    (DType.Float16, DType.BFloat16): DType.BFloat16,
    (DType.Float32, DType.Float32): DType.Float32,
    (DType.Float32, DType.Float64): DType.Float64,
    (DType.Float32, DType.Complex64): DType.Complex64,
    (DType.Float32, DType.Complex128): DType.Complex128,
    (DType.Float32, DType.BFloat16): DType.Float32,
    (DType.Float64, DType.Float64): DType.Float64,
    (DType.Float64, DType.Complex64): DType.Complex128,
    (DType.Float64, DType.Complex128): DType.Complex128,
    (DType.Float64, DType.BFloat16): DType.Float64,
    (DType.Complex64, DType.Complex64): DType.Complex64,
    (DType.Complex64, DType.Complex128): DType.Complex128,
    (DType.Complex64, DType.BFloat16): DType.Complex64,
    (DType.Complex128, DType.Complex128): DType.Complex128,
    (DType.Complex128, DType.BFloat16): DType.Complex128,
    (DType.BFloat16, DType.BFloat16): DType.BFloat16,
    (DType.Float8E4M3B11FNUZ, DType.Float8E4M3B11FNUZ): DType.Float8E4M3B11FNUZ,
    (DType.Float8E4M3B11FNUZ, DType.Bool): DType.Float8E4M3B11FNUZ,
    (DType.Float8E4M3B11FNUZ, DType.UInt8): DType.Float8E4M3B11FNUZ,
    (DType.Float8E4M3B11FNUZ, DType.Int8): DType.Float8E4M3B11FNUZ,
    (DType.Float8E4M3B11FNUZ, DType.Int16): DType.Float32,
    (DType.Float8E4M3B11FNUZ, DType.Int32): DType.Float64,
    (DType.Float8E4M3B11FNUZ, DType.Int64): DType.Float64,
    (DType.Float8E4M3B11FNUZ, DType.Float16): DType.Float16,
    (DType.Float8E4M3B11FNUZ, DType.BFloat16): DType.BFloat16,
    (DType.Float8E4M3B11FNUZ, DType.Float32): DType.Float32,
    (DType.Float8E4M3B11FNUZ, DType.Float64): DType.Float64,
    (DType.Float8E4M3B11FNUZ, DType.Complex64): DType.Complex64,
    (DType.Float8E4M3B11FNUZ, DType.Complex128): DType.Complex128,
    (DType.Float8E4M3FN, DType.Float8E4M3FN): DType.Float8E4M3FN,
    (DType.Float8E4M3FN, DType.Bool): DType.Float8E4M3FN,
    (DType.Float8E4M3FN, DType.UInt8): DType.Float8E4M3FN,
    (DType.Float8E4M3FN, DType.Int8): DType.Float8E4M3FN,
    (DType.Float8E4M3FN, DType.Int16): DType.Float32,
    (DType.Float8E4M3FN, DType.Int32): DType.Float64,
    (DType.Float8E4M3FN, DType.Int64): DType.Float64,
    (DType.Float8E4M3FN, DType.Float16): DType.Float16,
    (DType.Float8E4M3FN, DType.BFloat16): DType.BFloat16,
    (DType.Float8E4M3FN, DType.Float32): DType.Float32,
    (DType.Float8E4M3FN, DType.Float64): DType.Float64,
    (DType.Float8E4M3FN, DType.Complex64): DType.Complex64,
    (DType.Float8E4M3FN, DType.Complex128): DType.Complex128,
    (DType.Float8E4M3FNUZ, DType.Float8E4M3FNUZ): DType.Float8E4M3FNUZ,
    (DType.Float8E4M3FNUZ, DType.Bool): DType.Float8E4M3FNUZ,
    (DType.Float8E4M3FNUZ, DType.UInt8): DType.Float8E4M3FNUZ,
    (DType.Float8E4M3FNUZ, DType.Int8): DType.Float8E4M3FNUZ,
    (DType.Float8E4M3FNUZ, DType.Int16): DType.Float32,
    (DType.Float8E4M3FNUZ, DType.Int32): DType.Float64,
    (DType.Float8E4M3FNUZ, DType.Int64): DType.Float64,
    (DType.Float8E4M3FNUZ, DType.Float16): DType.Float16,
    (DType.Float8E4M3FNUZ, DType.BFloat16): DType.BFloat16,
    (DType.Float8E4M3FNUZ, DType.Float32): DType.Float32,
    (DType.Float8E4M3FNUZ, DType.Float64): DType.Float64,
    (DType.Float8E4M3FNUZ, DType.Complex64): DType.Complex64,
    (DType.Float8E4M3FNUZ, DType.Complex128): DType.Complex128,
    (DType.Float8E5M2, DType.Float8E5M2): DType.Float8E5M2,
    (DType.Float8E5M2, DType.Bool): DType.Float8E5M2,
    (DType.Float8E5M2, DType.UInt8): DType.Float8E5M2,
    (DType.Float8E5M2, DType.Int8): DType.Float8E5M2,
    (DType.Float8E5M2, DType.Int16): DType.Float32,
    (DType.Float8E5M2, DType.Int32): DType.Float64,
    (DType.Float8E5M2, DType.Int64): DType.Float64,
    (DType.Float8E5M2, DType.Float16): DType.Float16,
    (DType.Float8E5M2, DType.BFloat16): DType.BFloat16,
    (DType.Float8E5M2, DType.Float32): DType.Float32,
    (DType.Float8E5M2, DType.Float64): DType.Float64,
    (DType.Float8E5M2, DType.Complex64): DType.Complex64,
    (DType.Float8E5M2, DType.Complex128): DType.Complex128,
    (DType.Float8E5M2FNUZ, DType.Float8E5M2FNUZ): DType.Float8E5M2FNUZ,
    (DType.Float8E5M2FNUZ, DType.Bool): DType.Float8E5M2FNUZ,
    (DType.Float8E5M2FNUZ, DType.UInt8): DType.Float8E5M2FNUZ,
    (DType.Float8E5M2FNUZ, DType.Int8): DType.Float8E5M2FNUZ,
    (DType.Float8E5M2FNUZ, DType.Int16): DType.Float32,
    (DType.Float8E5M2FNUZ, DType.Int32): DType.Float64,
    (DType.Float8E5M2FNUZ, DType.Int64): DType.Float64,
    (DType.Float8E5M2FNUZ, DType.Float16): DType.Float16,
    (DType.Float8E5M2FNUZ, DType.BFloat16): DType.BFloat16,
    (DType.Float8E5M2FNUZ, DType.Float32): DType.Float32,
    (DType.Float8E5M2FNUZ, DType.Float64): DType.Float64,
    (DType.Float8E5M2FNUZ, DType.Complex64): DType.Complex64,
    (DType.Float8E5M2FNUZ, DType.Complex128): DType.Complex128,
    (DType.String, DType.String): DType.String,
}


def _clamp_x64(dtype: DType) -> DType:
    """Clamps x64 types to x32 if jax_enable_x64 is false.

    Args:
        dtype (DType): The dtype parameter.

    Returns:
        DType: Result.
    """
    if config.jax_enable_x64:
        return dtype
    _downcast = {
        DType.Float64: DType.Float32,
        DType.Int64: DType.Int32,
        DType.Complex128: DType.Complex64,
    }
    return _downcast.get(dtype, dtype)


def promote_types(dtype1: DType, dtype2: DType) -> DType:
    """Determine the resulting DType when operating on two tensors of dtype1 and dtype2.

    Args:
        dtype1 (DType): The dtype1 parameter.
        dtype2 (DType): The dtype2 parameter.

    Returns:
        DType: Result.

    Raises:
        DTypePromotionError: An exception.
    """
    if dtype1 == dtype2:
        return _clamp_x64(dtype1)

    # Order doesn't matter for promotion
    res = _PROMOTION_TABLE.get((dtype1, dtype2))
    if res is None:
        res = _PROMOTION_TABLE.get((dtype2, dtype1))

    if res is None:
        msg = f"Cannot promote types: {dtype1} and {dtype2}"
        raise DTypePromotionError(msg)

    # Preserve exact input type object if equal (for custom dtype instances)
    if res == dtype1:
        res = dtype1
    elif res == dtype2:
        res = dtype2

    return _clamp_x64(res)
