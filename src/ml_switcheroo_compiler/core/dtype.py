# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module dtype.py."""

"""DType enums for the ml-switcheroo compiler."""

from enum import Enum


class DType(Enum):
    """Standard data types supported by the compiler."""

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
