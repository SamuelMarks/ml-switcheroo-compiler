# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Module dtype.py."""

"""DType enums for the ml-switcheroo compiler."""

from enum import Enum


class DType(Enum):
    """Standard data types supported by the compiler."""

    Float64: object = "float64"
    Float32: object = "float32"
    Float16: object = "float16"
    BFloat16: object = "bfloat16"
    Float8E4M3B11FNUZ: object = "float8_e4m3b11fnuz"
    Float8E4M3FN: object = "float8_e4m3fn"
    Float8E4M3FNUZ: object = "float8_e4m3fnuz"
    Float8E5M2: object = "float8_e5m2"
    Float8E5M2FNUZ: object = "float8_e5m2fnuz"
    Complex64: object = "complex64"
    Complex128: object = "complex128"
    Int64: object = "int64"
    Int32: object = "int32"
    Int16: object = "int16"
    Int8: object = "int8"
    Int4: object = "int4"
    UInt64: object = "uint64"
    UInt32: object = "uint32"
    UInt16: object = "uint16"
    UInt8: object = "uint8"
    UInt4: object = "uint4"
    Bool: object = "bool"
    String: object = "string"
    Object: object = "object"


class QuantDType(Enum):
    """Quantized data types supported by the compiler."""

    QInt8: object = "qint8"
    QUInt8: object = "quint8"
    QInt4: object = "qint4"


# Type aliases
bfloat16: object = DType.BFloat16
float8_e4m3b11fnuz: object = DType.Float8E4M3B11FNUZ
float8_e4m3fn: object = DType.Float8E4M3FN
float8_e4m3fnuz: object = DType.Float8E4M3FNUZ
float8_e5m2: object = DType.Float8E5M2
float8_e5m2fnuz: object = DType.Float8E5M2FNUZ
uint16: object = DType.UInt16
uint32: object = DType.UInt32
uint64: object = DType.UInt64
int4: object = DType.Int4
int8: object = DType.Int8
int16: object = DType.Int16
float16: object = DType.Float16
float64: object = DType.Float64
cdouble: object = DType.Complex128
csingle: object = DType.Complex64
double: object = DType.Float64
single: object = DType.Float32
bool_: object = DType.Bool
int_: object = DType.Int64
float_: object = DType.Float64
complex_: object = DType.Complex128
object_: object = DType.Object

# Type categories
floating: object = (
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
complexfloating: object = (DType.Complex64, DType.Complex128)
inexact: object = floating + complexfloating
signedinteger: object = (DType.Int64, DType.Int32, DType.Int16, DType.Int8, DType.Int4)
unsignedinteger: object = (DType.UInt64, DType.UInt32, DType.UInt16, DType.UInt8, DType.UInt4)
integer: object = signedinteger + unsignedinteger
number: object = inexact + integer
generic: object = number + (DType.Bool, DType.String, DType.Object)
