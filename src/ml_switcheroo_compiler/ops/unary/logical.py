"""Module docstring."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op

from .base import UnaryMathOp


@register_op("BitwiseNot")
class BitwiseNot(UnaryMathOp):
    """Computes bitwise NOT element-wise."""

    op_name = "BitwiseNot"
    np_op_name = "bitwise_not"


@register_op("Isfinite")
class Isfinite(UnaryMathOp):
    """Tests element-wise for finiteness (not infinity or NaN)."""

    op_name = "Isfinite"
    np_op_name = "isfinite"


@register_op("Isinf")
class Isinf(UnaryMathOp):
    """Tests element-wise for positive or negative infinity."""

    op_name = "Isinf"
    np_op_name = "isinf"


@register_op("Isnan")
class Isnan(UnaryMathOp):
    """Tests element-wise for NaN (Not a Number)."""

    op_name = "Isnan"
    np_op_name = "isnan"


@register_op("LogicalNot")
class LogicalNot(UnaryMathOp):
    """Computes the truth value of NOT x element-wise."""

    op_name = "LogicalNot"
    np_op_name = "logical_not"


@register_op("BitwiseCount")
class BitwiseCount(UnaryMathOp):
    """Computes the number of 1-bits in the binary representation of x."""

    op_name = "BitwiseCount"
    np_op_name = "bitwise_count"


@register_op("IsNonDecreasing")
class IsNonDecreasing(UnaryMathOp):
    """IsNonDecreasing operation."""

    op_name = "IsNonDecreasing"


@register_op("IsStrictlyIncreasing")
class IsStrictlyIncreasing(UnaryMathOp):
    """IsStrictlyIncreasing operation."""

    op_name = "IsStrictlyIncreasing"


@register_op("Packbits")
class Packbits(OpDef):
    """Packbits operator definition."""

    op_name = "Packbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Unpackbits")
class Unpackbits(OpDef):
    """Unpackbits operator definition."""

    op_name = "Unpackbits"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Clz")
class Clz(UnaryMathOp):
    """Count leading zeros."""

    op_name = "Clz"


@register_op("PopulationCount")
class PopulationCount(UnaryMathOp):
    """Population count."""

    op_name = "PopulationCount"


@register_op("BitcastConvertType")
class BitcastConvertType(UnaryMathOp):
    """Bitcast convert type."""

    op_name = "BitcastConvertType"


@register_op("ReducePrecision")
class ReducePrecision(UnaryMathOp):
    """Reduce precision."""

    op_name = "ReducePrecision"
