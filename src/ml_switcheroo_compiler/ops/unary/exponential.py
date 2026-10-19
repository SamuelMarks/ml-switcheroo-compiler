"""Implementation of exponential operations and utilities for the ops subsystem (`ml_switcheroo_compiler.ops.unary.exponential`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ml_switcheroo_compiler.ops.base import register_op

from .base import UnaryMathOp

if TYPE_CHECKING:
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.ir.core import IRNode


@register_op("Exp")
class Exp(UnaryMathOp):
    """Compute the exponential of all elements in the input."""

    op_name = "Exp"


@register_op("Log")
class Log(UnaryMathOp):
    """Compute the natural logarithm element-wise."""

    op_name = "Log"


@register_op("Exp2")
class Exp2(UnaryMathOp):
    """Compute 2**x element-wise."""

    op_name = "Exp2"
    np_op_name = "exp2"


@register_op("Expm1")
class Expm1(UnaryMathOp):
    """Compute exp(x) - 1 element-wise."""

    op_name = "Expm1"
    np_op_name = "expm1"


@register_op("Log10")
class Log10(UnaryMathOp):
    """Compute the base-10 logarithm element-wise."""

    op_name = "Log10"
    np_op_name = "log10"


@register_op("Log1P")
class Log1P(UnaryMathOp):
    """Compute natural logarithm of 1 + x element-wise."""

    op_name = "Log1P"
    np_op_name = "log1p"


@register_op("Log2")
class Log2(UnaryMathOp):
    """Compute the base-2 logarithm element-wise."""

    op_name = "Log2"
    np_op_name = "log2"


@register_op("Logit")
class Logit(UnaryMathOp):
    """Compute the logit of a tensor element-wise."""

    op_name = "Logit"


@register_op("NanToNum")
class NanToNum(UnaryMathOp):
    """Replace NaN, positive infinity, and negative infinity values."""

    op_name = "NanToNum"

    def __call__(
        self,
        x: Tensor | IRNode | int | float | str,
        **kwargs: int | float | str | bool | None,
    ) -> Tensor:
        """Call NanToNum, filtering out the copy kwarg.

        Args:
            x (Tensor | IRNode | int | float | str): The input parameter.
            **kwargs (int | float | str | bool | None): Keyword args.

        Returns:
            Tensor: Result.
        """
        clean_kwargs = {k: v for k, v in kwargs.items() if k != "copy"}
        return super().__call__(x, **clean_kwargs)


@register_op("ZeroFraction")
class ZeroFraction(UnaryMathOp):
    """ZeroFraction operation."""

    op_name = "ZeroFraction"
