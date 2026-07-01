"""Defines special unary operations for the ML Switcheroo framework, including Cast,.

Bitcast, and Frexp
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Cast")
class Cast(OpDef):
    """An operation that casts an input array to a specified data type."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            dtype (object, optional): The target data type.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x


@register_op("Bitcast")
class Bitcast(Cast):
    """An operation that bitcasts an input array to a specified data type without copying."""


@register_op("Frexp")
class Frexp(OpDef):
    """An operation that decomposes a floating-point array into mantissa and exponent."""

    def infer_shape(self, x: object, dtype: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            dtype (object, optional): The target data type.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x
