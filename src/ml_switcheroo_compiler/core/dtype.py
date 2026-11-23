"""DType enums for the ml-switcheroo compiler."""

from __future__ import annotations

from enum import Enum

from ml_switcheroo_ir import (
    Codec,
    CodecError,
    Eq,
    Err,
    HashableWitness,
    Ok,
    Ord,
    Ordering,
    Pretty,
    PythonExprEmit,
    Result,
)

from ml_switcheroo_compiler.core.errors import DTypeValidationError as DTypeValidationError

_DTYPE_ALIASES: dict[str, str] = {
    "f64": "float64",
    "double": "float64",
    "f32": "float32",
    "single": "float32",
    "float": "float32",
    "f16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "i64": "int64",
    "long": "int64",
    "int": "int64",
    "i32": "int32",
    "i16": "int16",
    "short": "int16",
    "i8": "int8",
    "i4": "int4",
    "i2": "int2",
    "u64": "uint64",
    "u32": "uint32",
    "u16": "uint16",
    "u8": "uint8",
    "byte": "uint8",
    "u4": "uint4",
    "c64": "complex64",
    "csingle": "complex64",
    "c128": "complex128",
    "cdouble": "complex128",
    "complex": "complex128",
    "bool_": "bool",
    "boolean": "bool",
    "str": "string",
}


class DType(str, Enum):
    """Canonical tensor element data types across ML frameworks and hardware dialects.

    Supports both lowercase canonical identifiers (e.g. ``DType.float32``) and
    PascalCase aliases (e.g. ``DType.Float32``) natively as enum members.
    """

    float64 = "float64"
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float8_e4m3b11fnuz = "float8_e4m3b11fnuz"
    float8_e4m3fn = "float8_e4m3fn"
    fp8_e4m3fn = "fp8_e4m3fn"
    float8_e4m3fnuz = "fp8_e4m3fnuz"
    fp8_e4m3fnuz = "fp8_e4m3fnuz"
    float8_e5m2 = "float8_e5m2"
    fp8_e5m2 = "fp8_e5m2"
    float8_e5m2fnuz = "fp8_e5m2fnuz"
    fp8_e5m2fnuz = "fp8_e5m2fnuz"
    complex64 = "complex64"
    complex128 = "complex128"
    int64 = "int64"
    int32 = "int32"
    int16 = "int16"
    int8 = "int8"
    int4 = "int4"
    int2 = "int2"
    uint64 = "uint64"
    uint32 = "uint32"
    uint16 = "uint16"
    uint8 = "uint8"
    uint4 = "uint4"
    qint8 = "qint8"
    quint8 = "quint8"
    qint4 = "qint4"
    bool = "bool"
    string = "string"
    object_ = "object"

    # PascalCase canonical aliases
    Float64 = "float64"
    Float32 = "float32"
    Float16 = "float16"
    BFloat16 = "bfloat16"
    Float8E4M3B11FNUZ = "float8_e4m3b11fnuz"
    Float8E4M3FN = "float8_e4m3fn"
    Float8E4M3FNUZ = "fp8_e4m3fnuz"
    Float8E5M2 = "float8_e5m2"
    Float8E5M2FNUZ = "fp8_e5m2fnuz"
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

    @classmethod
    def from_str(cls, val: str) -> DType:
        """Parse a string or framework-specific name into a canonical ``DType``.

        Args:
            val (str): String representation of the data type (case-insensitive).

        Returns:
            DType: Corresponding canonical ``DType`` enum member.

        Raises:
            ValueError: If ``val`` cannot be resolved to a known ``DType``.
        """
        clean = val.strip().lower().split(".")[-1]
        mapped = _DTYPE_ALIASES.get(clean, clean)
        for member in cls:
            if member.value == mapped or member.name.lower() == clean:
                return member
        raise ValueError(f"Unknown or unsupported DType representation: {val!r}")


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
            return False, (
                f"Data type '{dtype_str}' is not supported for transcendental operation '{op_name}'. "
                "Floating-point or complex dtype required."
            )
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


def parse_dtype_result(val: DType | str) -> Result[DTypeValidationError, DType]:
    """Parse a ``DType`` or string alias into a total ``Result`` without raising exceptions.

    Args:
        val (DType | str): Canonical ``DType`` member or string alias.

    Returns:
        Result[DTypeValidationError, DType]: ``Ok(DType)`` on success or ``Err(DTypeValidationError)`` on unknown input.
    """
    if isinstance(val, DType):
        return Ok(val)
    try:
        return Ok(DType.from_str(val))
    except ValueError as exc:
        return Err(DTypeValidationError(str(exc)))


_DTYPE_CANONICAL_ORDER: tuple[DType, ...] = tuple(DType)
_DTYPE_RANK_MAP: dict[DType, int] = {dt: idx for idx, dt in enumerate(_DTYPE_CANONICAL_ORDER)}


def _compare_dtypes(lhs: DType, rhs: DType) -> Ordering:
    """Compare two ``DType`` values deterministically according to canonical declaration rank.

    Args:
        lhs (DType): Left-hand ``DType``.
        rhs (DType): Right-hand ``DType``.

    Returns:
        Ordering: ``Ordering.LT``, ``Ordering.EQ``, or ``Ordering.GT``.
    """
    rank_l: int = _DTYPE_RANK_MAP[lhs]
    rank_r: int = _DTYPE_RANK_MAP[rhs]
    if rank_l < rank_r:
        return Ordering.LT
    if rank_l > rank_r:
        return Ordering.GT
    return Ordering.EQ


def _emit_dtype_python_expr(dt: DType, module_prefix: str) -> str:
    """Emit a deterministic Python constructor expression for ``dt``.

    Args:
        dt (DType): Canonical ``DType`` enum member.
        module_prefix (str): Optional module qualifier prefix.

    Returns:
        str: Valid Python attribute expression referencing ``dt``.
    """
    if module_prefix:
        return f"{module_prefix}.DType.{dt.name}"
    return f"DType.{dt.name}"


def _decode_dtype_wire(wire: str) -> Result[CodecError, DType]:
    """Decode a wire string representation into a canonical ``DType``.

    Args:
        wire (str): Serialized ``DType`` string or alias.

    Returns:
        Result[CodecError, DType]: ``Ok(DType)`` or ``Err(CodecError)``.
    """
    parsed = parse_dtype_result(wire)
    if isinstance(parsed, Err):
        return Err(
            CodecError(
                path=("dtype",),
                expected="DType",
                actual_description=wire,
                message=str(parsed.error),
            )
        )
    return Ok(parsed.value)


COMPILER_DTYPE_EQ: Eq[DType] = Eq(eq=lambda a, b: a == b)
COMPILER_DTYPE_HASH: HashableWitness[DType] = HashableWitness(
    eq_witness=COMPILER_DTYPE_EQ,
    hash_value=lambda dt: hash(dt.value),
)
COMPILER_DTYPE_ORD: Ord[DType] = Ord(
    eq_witness=COMPILER_DTYPE_EQ,
    compare=_compare_dtypes,
)
COMPILER_DTYPE_PRETTY: Pretty[DType] = Pretty(render=lambda dt: dt.value)
COMPILER_DTYPE_PYTHON_EMIT: PythonExprEmit[DType] = PythonExprEmit(emit_python=_emit_dtype_python_expr)
COMPILER_DTYPE_STR_CODEC: Codec[str, DType] = Codec(
    encode=lambda dt: dt.value,
    decode=_decode_dtype_wire,
)
