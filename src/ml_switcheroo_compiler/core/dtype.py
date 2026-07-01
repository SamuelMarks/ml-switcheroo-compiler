"""DType enums for the ml-switcheroo compiler."""

from enum import Enum


class DType(Enum):
    """Standard data types supported by the compiler."""

    Float64 = "float64"
    Float32 = "float32"
    Float16 = "float16"
    BFloat16 = "bfloat16"
    Complex64 = "complex64"
    Complex128 = "complex128"
    Int64 = "int64"
    Int32 = "int32"
    UInt32 = "uint32"
    Int16 = "int16"
    Int8 = "int8"
    UInt8 = "uint8"
    Bool = "bool"
    String = "string"


class QuantDType(Enum):
    """Quantized data types supported by the compiler."""

    QInt8 = "qint8"
    QUInt8 = "quint8"
    QInt4 = "qint4"


# Type categories
floating = (DType.Float64, DType.Float32, DType.Float16, DType.BFloat16)
complexfloating = (DType.Complex64, DType.Complex128)
inexact = floating + complexfloating
signedinteger = (DType.Int64, DType.Int32, DType.Int16, DType.Int8)
unsignedinteger = (DType.UInt32, DType.UInt8)
integer = signedinteger + unsignedinteger
number = inexact + integer
generic = number + (DType.Bool, DType.String)
