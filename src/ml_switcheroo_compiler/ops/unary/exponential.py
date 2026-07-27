"""Core abstractions and logic definitions for exponential.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Exp")
class Exp(UnaryMathOp):
    """Computes the exponential of all elements in the input."""

    op_name = "Exp"


@register_op("Log")
class Log(UnaryMathOp):
    """Computes the natural logarithm element-wise."""

    op_name = "Log"


@register_op("Exp2")
class Exp2(UnaryMathOp):
    """Computes 2**x element-wise."""

    op_name = "Exp2"
    np_op_name = "exp2"


@register_op("Expm1")
class Expm1(UnaryMathOp):
    """Computes exp(x) - 1 element-wise."""

    op_name = "Expm1"
    np_op_name = "expm1"


@register_op("Log10")
class Log10(UnaryMathOp):
    """Computes the base-10 logarithm element-wise."""

    op_name = "Log10"
    np_op_name = "log10"


@register_op("Log1P")
class Log1P(UnaryMathOp):
    """Computes natural logarithm of 1 + x element-wise."""

    op_name = "Log1P"
    np_op_name = "log1p"


@register_op("Log2")
class Log2(UnaryMathOp):
    """Computes the base-2 logarithm element-wise."""

    op_name = "Log2"
    np_op_name = "log2"


@register_op("Logit")
class Logit(UnaryMathOp):
    """Computes the logit of a tensor element-wise."""

    op_name = "Logit"


@register_op("NanToNum")
class NanToNum(UnaryMathOp):
    """Replaces NaN, positive infinity, and negative infinity values."""

    op_name = "NanToNum"

    def __call__(self, x: object, **kwargs: object) -> object:
        """Call NanToNum, filtering out the copy kwarg."""
        kwargs.pop("copy", None)
        return super().__call__(x, **kwargs)


@register_op("ZeroFraction")
class ZeroFraction(UnaryMathOp):
    """ZeroFraction operation."""

    op_name = "ZeroFraction"
