"""DType enums for the ml-switcheroo compiler."""

from __future__ import annotations

import typing
from enum import Enum

from ml_switcheroo_ir.types import DType as _IRDType

from ml_switcheroo_compiler.core.errors import DTypeValidationError as DTypeValidationError

if typing.TYPE_CHECKING:

    class DType(str, Enum):
        """Canonical DType enumeration with lowercase and PascalCase members."""

        float64 = "float64"
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"
        float8_e4m3b11fnuz = "float8_e4m3b11fnuz"
        float8_e4m3fn = "float8_e4m3fn"
        float8_e4m3fnuz = "float8_e4m3fnuz"
        fp8_e4m3fnuz = "float8_e4m3fnuz"
        float8_e5m2 = "float8_e5m2"
        float8_e5m2fnuz = "float8_e5m2fnuz"
        fp8_e5m2fnuz = "float8_e5m2fnuz"
        complex64 = "complex64"
        complex128 = "complex128"
        int64 = "int64"
        int32 = "int32"
        int16 = "int16"
        int8 = "int8"
        int4 = "int4"
        uint64 = "uint64"
        uint32 = "uint32"
        uint16 = "uint16"
        uint8 = "uint8"
        uint4 = "uint4"
        bool = "bool"
        string = "string"
        object_ = "object"

        Float64 = "float64"
        Float32 = "float32"
        Float16 = "float16"
        BFloat16 = "bfloat16"
        Float8E4M3B11FNUZ = "float8_e4m3b11fnuz"
        Float8E4M3FN = "float8_e4m3fn"
        Float8E4M3FNUZ = "float8_e4m3fnuz"
        Float8E5M2 = "float8_e5m2"
        Float8E5M2FNUZ = "float8_e5m2fnuz"
        Complex64 = "complex64"
        Complex128 = "complex128"
        Int64 = "int64"
        Int32 = "int32"
        Int16 = "int16"
        Int8 = "int8"
        Int4 = "int4"
        UInt64 = "uint64"
        UInt32 = "uint32"
        UInt16 = "uint16"
        UInt8 = "uint8"
        UInt4 = "uint4"
        Bool = "bool"
        String = "string"
        Object = "object"


class _MutableDTypeClsProtocol(typing.Protocol):
    """Protocol for attaching backward-compatible PascalCase aliases to `DType`."""

    Float64: _IRDType
    Float32: _IRDType
    Float16: _IRDType
    BFloat16: _IRDType
    Float8E4M3B11FNUZ: _IRDType
    Float8E4M3FN: _IRDType
    float8_e4m3fnuz: _IRDType
    float8_e5m2fnuz: _IRDType
    Float8E4M3FNUZ: _IRDType
    Float8E5M2: _IRDType
    Float8E5M2FNUZ: _IRDType
    Complex64: _IRDType
    Complex128: _IRDType
    Int64: _IRDType
    Int32: _IRDType
    Int16: _IRDType
    Int8: _IRDType
    Int4: _IRDType
    UInt64: _IRDType
    UInt32: _IRDType
    UInt16: _IRDType
    UInt8: _IRDType
    UInt4: _IRDType
    Bool: _IRDType
    String: _IRDType
    Object: _IRDType
    object_: _IRDType


def _is_mutable_dtype_cls(cls_obj: type[_IRDType]) -> typing.TypeGuard[_MutableDTypeClsProtocol]:
    """Type guard narrowing `type[_IRDType]` to `_MutableDTypeClsProtocol`.

    Args:
        cls_obj (type[_IRDType]): Canonical IR DType enum class.

    Returns:
        bool: True if cls_obj is a class.
    """
    return isinstance(cls_obj, type)


def _is_dtype_enum_cls(cls_obj: type[_IRDType]) -> typing.TypeGuard[type[DType]]:
    """Type guard narrowing `type[_IRDType]` to `type[DType]`.

    Args:
        cls_obj (type[_IRDType]): Canonical IR DType enum class.

    Returns:
        bool: True if cls_obj is a class.
    """
    return isinstance(cls_obj, type)


def _setup_runtime_dtype() -> type[DType]:
    """Configure runtime PascalCase attribute aliases on the canonical IR `DType` enum.

    Returns:
        type[DType]: Configured `DType` enumeration class.
    """
    raw_cls = _IRDType
    assert _is_mutable_dtype_cls(raw_cls)
    raw_cls.Float64 = _IRDType.float64
    raw_cls.Float32 = _IRDType.float32
    raw_cls.Float16 = _IRDType.float16
    raw_cls.BFloat16 = _IRDType.bfloat16
    raw_cls.Float8E4M3B11FNUZ = _IRDType.float8_e4m3b11fnuz
    raw_cls.Float8E4M3FN = _IRDType.float8_e4m3fn
    raw_cls.float8_e4m3fnuz = _IRDType.fp8_e4m3fnuz
    raw_cls.float8_e5m2fnuz = _IRDType.fp8_e5m2fnuz
    raw_cls.Float8E4M3FNUZ = _IRDType.fp8_e4m3fnuz
    raw_cls.Float8E5M2 = _IRDType.float8_e5m2
    raw_cls.Float8E5M2FNUZ = _IRDType.fp8_e5m2fnuz
    raw_cls.Complex64 = _IRDType.complex64
    raw_cls.Complex128 = _IRDType.complex128
    raw_cls.Int64 = _IRDType.int64
    raw_cls.Int32 = _IRDType.int32
    raw_cls.Int16 = _IRDType.int16
    raw_cls.Int8 = _IRDType.int8
    raw_cls.Int4 = _IRDType.int4
    raw_cls.UInt64 = _IRDType.uint64
    raw_cls.UInt32 = _IRDType.uint32
    raw_cls.UInt16 = _IRDType.uint16
    raw_cls.UInt8 = _IRDType.uint8
    raw_cls.UInt4 = _IRDType.uint4
    raw_cls.Bool = _IRDType.bool
    raw_cls.String = _IRDType.string
    raw_cls.Object = _IRDType["object"]
    raw_cls.object_ = _IRDType["object"]
    ret_cls = _IRDType
    assert _is_dtype_enum_cls(ret_cls)
    return ret_cls


if not typing.TYPE_CHECKING:  # pragma: no cover
    DType = _setup_runtime_dtype()


class QuantDType(Enum):
    """Quantized data types supported by the compiler."""

    QInt8 = "qint8"
    QUInt8 = "quint8"
    QInt4 = "qint4"


# Type aliases
bfloat16 = DType.BFloat16
float8_e4m3b11fnuz = DType.Float8E4M3B11FNUZ
float8_e4m3fn = DType.Float8E4M3FN
float8_e4m3fnuz = DType.Float8E4M3FNUZ
float8_e5m2 = DType.Float8E5M2
float8_e5m2fnuz = DType.Float8E5M2FNUZ
uint16 = DType.UInt16
uint32 = DType.UInt32
uint64 = DType.UInt64
int4 = DType.Int4
int8 = DType.Int8
int16 = DType.Int16
float16 = DType.Float16
float64 = DType.Float64
cdouble = DType.Complex128
csingle = DType.Complex64
double = DType.Float64
single = DType.Float32
bool_ = DType.Bool
int_ = DType.Int64
float_ = DType.Float64
complex_ = DType.Complex128
object_ = DType.Object

# Type categories
floating = (
    DType.Float64,
    DType.Float32,
    DType.Float16,
    DType.BFloat16,
    DType.Float8E4M3B11FNUZ,
    DType.Float8E4M3FN,
    DType.Float8E4M3FNUZ,
    DType.Float8E5M2,
    DType.Float8E5M2FNUZ,
)
complexfloating = (DType.Complex64, DType.Complex128)
inexact = floating + complexfloating
signedinteger = (DType.Int64, DType.Int32, DType.Int16, DType.Int8, DType.Int4)
unsignedinteger = (DType.UInt64, DType.UInt32, DType.UInt16, DType.UInt8, DType.UInt4)
integer = signedinteger + unsignedinteger
number = inexact + integer
generic = number + (DType.Bool, DType.String, DType.Object)


def validate_dtype_for_op(op_name: str, dtype: DType | str) -> tuple[bool, str | None]:
    """Validate that a data type is legally applicable to a mathematical operator.

    Rejects quantized, integer, boolean, and non-numeric types for transcendental
    operations (e.g., 'sin', 'exp', 'log', 'cos', 'sqrt', 'cholesky', 'linalg_inv', 'inv').

    Args:
        op_name (str): Operation or function name.
        dtype (DType | str): Data type or DType enum to validate.

    Returns:
        tuple[bool, str | None]: Tuple of (is_valid, optional_error_message).
    """
    dtype_str: str = dtype.value if isinstance(dtype, DType) else str(dtype)
    clean_op: str = op_name.lower().split(".")[-1]
    transcendental_ops: tuple[str, ...] = (
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sigmoid",
        "tanh",
        "cholesky",
        "linalg_inv",
        "inv",
    )
    if clean_op in transcendental_ops:
        clean_dt: str = dtype_str.lower()
        non_float_prefixes: tuple[str, ...] = (
            "int",
            "uint",
            "qint",
            "quint",
            "bool",
            "string",
            "object",
        )
        if any(clean_dt.startswith(p) for p in non_float_prefixes):
            return False, (f"Data type '{dtype_str}' is not supported for transcendental operation '{op_name}'. Floating-point or complex dtype required.")
    return True, None


def check_dtype_for_op(op_name: str, dtype: DType | str) -> None:
    """Enforce data type compatibility for an operation, raising DTypeValidationError if rejected.

    Args:
        op_name (str): Operation or function name.
        dtype (DType | str): Data type or DType enum to validate.

    Raises:
        DTypeValidationError: If the data type is incompatible with the operation.
    """
    valid: bool
    err: str | None
    valid, err = validate_dtype_for_op(op_name, dtype)
    if not valid:
        raise DTypeValidationError(err or f"Invalid dtype '{dtype}' for op '{op_name}'")
