# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module dtype.py."""

from __future__ import annotations

"""DType enums for the ml-switcheroo compiler."""

from enum import Enum

from ml_switcheroo_ir.types import DType as DType

# Re-export canonical DType and add backward-compatible capitalized attribute aliases
DType.Float64 = DType.float64
DType.Float32 = DType.float32
DType.Float16 = DType.float16
DType.BFloat16 = DType.bfloat16
DType.Float8E4M3B11FNUZ = DType.float8_e4m3b11fnuz
DType.Float8E4M3FN = DType.float8_e4m3fn
DType.Float8E4M3FNUZ = getattr(DType, "float8_e4m3fnuz", DType.fp8_e4m3fnuz)
DType.Float8E5M2 = DType.float8_e5m2
DType.Float8E5M2FNUZ = getattr(DType, "float8_e5m2fnuz", DType.fp8_e5m2fnuz)
DType.Complex64 = DType.complex64
DType.Complex128 = DType.complex128
DType.Int64 = DType.int64
DType.Int32 = DType.int32
DType.Int16 = DType.int16
DType.Int8 = DType.int8
DType.Int4 = DType.int4
DType.UInt64 = DType.uint64
DType.UInt32 = DType.uint32
DType.UInt16 = DType.uint16
DType.UInt8 = DType.uint8
DType.UInt4 = DType.uint4
DType.Bool = DType.bool
DType.String = DType.string
DType.Object = DType.object


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


class DTypeValidationError(TypeError):
    """Exception raised when an operand data type is invalid for an operation."""


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
    try:
        import importlib

        mod = importlib.import_module("ml_ecosystem_snapshots.compliance")
        fn = getattr(mod, "validate_dtype_for_op", None)
        if fn is not None:
            res: tuple[bool, str | None] = fn(op_name, dtype_str)
            return res
    except Exception:
        pass

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
