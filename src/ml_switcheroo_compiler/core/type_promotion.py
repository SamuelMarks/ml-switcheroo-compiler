"""Type promotion rules for ml-switcheroo."""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import DTypePromotionError


def _promote_complex(dtype1: DType, dtype2: DType) -> DType:
    """Promote complex types.

    Args:
        dtype1 (DType): The first dtype.
        dtype2 (DType): The second dtype.

    Returns:
        DType: The promoted dtype.
    """
    if DType.Complex128 in (dtype1, dtype2):
        return DType.Complex128
    if DType.Float64 in (dtype1, dtype2):
        return DType.Complex128
    return DType.Complex64


def _promote_float(dtype1: DType, dtype2: DType, rank1: int, rank2: int) -> DType:
    """Promote float types.

    Args:
        dtype1 (DType): The first dtype.
        dtype2 (DType): The second dtype.
        rank1 (int): The rank of the first dtype.
        rank2 (int): The rank of the second dtype.

    Returns:
        DType: The promoted dtype.
    """
    if DType.Float64 in (dtype1, dtype2):
        return DType.Float64
    if DType.Float32 in (dtype1, dtype2):
        return DType.Float32
    if (dtype1 == DType.Float16 and dtype2 == DType.BFloat16) or (
        dtype1 == DType.BFloat16 and dtype2 == DType.Float16
    ):
        return DType.Float32
    if rank1 > rank2:
        return dtype1
    return dtype2


def _promote_int(dtype1: DType, dtype2: DType, rank1: int, rank2: int) -> DType:
    """Promote int types.

    Args:
        dtype1 (DType): The first dtype.
        dtype2 (DType): The second dtype.
        rank1 (int): The rank of the first dtype.
        rank2 (int): The rank of the second dtype.

    Returns:
        DType: The promoted dtype.
    """
    if rank1 > rank2:
        return dtype1
    return dtype2


def promote_types(dtype1: DType, dtype2: DType) -> DType:
    """Determine the resulting DType when operating on two tensors of dtype1 and dtype2.

    Follows generalized NumPy promotion rules:
    - Booleans promote to integers
    - Integers promote to floats
    - Lower precision floats promote to higher precision
    - Reals promote to complex

    dtype1: The first data type
    dtype2: The second data type

    Returns:
    The resulting promoted DType

    Raises:
    DTypePromotionError: If the types cannot be safely promoted

    Args:
        dtype1 (DType): Argument dtype1
        dtype2 (DType): Argument dtype2
    """
    if dtype1 == dtype2:
        return dtype1

    _rank = {
        DType.Bool: 0,
        DType.UInt8: 1,
        DType.Int8: 2,
        DType.Int16: 3,
        DType.Int32: 4,
        DType.Int64: 5,
        DType.Float16: 6,
        DType.BFloat16: 7,
        DType.Float32: 8,
        DType.Float64: 9,
        DType.Complex64: 10,
        DType.Complex128: 11,
    }

    if dtype1 not in _rank or dtype2 not in _rank:
        msg = f"Cannot promote types: {dtype1} and {dtype2}"
        raise DTypePromotionError(msg)

    rank1 = _rank[dtype1]
    rank2 = _rank[dtype2]

    is_complex1 = dtype1 in (DType.Complex64, DType.Complex128)
    is_complex2 = dtype2 in (DType.Complex64, DType.Complex128)
    is_float1 = dtype1 in (DType.Float16, DType.BFloat16, DType.Float32, DType.Float64)
    is_float2 = dtype2 in (DType.Float16, DType.BFloat16, DType.Float32, DType.Float64)
    is_int1 = dtype1 in (DType.Int8, DType.Int16, DType.Int32, DType.Int64, DType.UInt8)
    is_int2 = dtype2 in (DType.Int8, DType.Int16, DType.Int32, DType.Int64, DType.UInt8)

    if is_complex1 or is_complex2:
        return _promote_complex(dtype1, dtype2)

    if is_float1 or is_float2:
        return _promote_float(dtype1, dtype2, rank1, rank2)

    if is_int1 or is_int2:
        return _promote_int(dtype1, dtype2, rank1, rank2)

    return dtype1
