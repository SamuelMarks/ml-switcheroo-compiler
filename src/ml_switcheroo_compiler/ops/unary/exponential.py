# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for exponential.py."""

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp


@register_op("Exp")
class Exp(UnaryMathOp):
    """Compute the exponential of all elements in the input."""

    op_name: object = "Exp"


@register_op("Log")
class Log(UnaryMathOp):
    """Compute the natural logarithm element-wise."""

    op_name: object = "Log"


@register_op("Exp2")
class Exp2(UnaryMathOp):
    """Compute 2**x element-wise."""

    op_name: object = "Exp2"
    np_op_name: object = "exp2"


@register_op("Expm1")
class Expm1(UnaryMathOp):
    """Compute exp(x) - 1 element-wise."""

    op_name: object = "Expm1"
    np_op_name: object = "expm1"


@register_op("Log10")
class Log10(UnaryMathOp):
    """Compute the base-10 logarithm element-wise."""

    op_name: object = "Log10"
    np_op_name: object = "log10"


@register_op("Log1P")
class Log1P(UnaryMathOp):
    """Compute natural logarithm of 1 + x element-wise."""

    op_name: object = "Log1P"
    np_op_name: object = "log1p"


@register_op("Log2")
class Log2(UnaryMathOp):
    """Compute the base-2 logarithm element-wise."""

    op_name: object = "Log2"
    np_op_name: object = "log2"


@register_op("Logit")
class Logit(UnaryMathOp):
    """Compute the logit of a tensor element-wise."""

    op_name: object = "Logit"


@register_op("NanToNum")
class NanToNum(UnaryMathOp):
    """Replace NaN, positive infinity, and negative infinity values."""

    op_name: object = "NanToNum"

    def __call__(self, x: object, **kwargs: object) -> object:
        """Call NanToNum, filtering out the copy kwarg.

        Args:
        x (object): The x parameter.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        kwargs.pop("copy", None)
        return super().__call__(x, **kwargs)


@register_op("ZeroFraction")
class ZeroFraction(UnaryMathOp):
    """ZeroFraction operation."""

    op_name: object = "ZeroFraction"
