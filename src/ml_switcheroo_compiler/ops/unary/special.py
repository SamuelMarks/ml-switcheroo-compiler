"""Defines special unary operations for the ML Switcheroo framework, including Cast,.

Bitcast, and Frexp
"""

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Cast")
class Cast(OpDef):
    """An operation that casts an input array to a specified data type."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The first input tensor.
            dtype (object): The dtype to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        import numpy as np

        dtype_val = kwargs.get("dtype")
        if isinstance(dtype_val, DType):
            dtype_val = dtype_val.value
        return np.array(x).astype(dtype_val)


@register_op("Bitcast")
class Bitcast(Cast):
    """An operation that bitcasts an input array to a specified data type without copying."""

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The first input tensor.
            dtype (object): The dtype to process.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        import numpy as np

        dtype_val = kwargs.get("dtype")
        if isinstance(dtype_val, DType):
            dtype_val = dtype_val.value
        return np.array(x).view(dtype_val)


@register_op("Frexp")
class Frexp(OpDef):
    """An operation that decomposes a floating-point array into mantissa and exponent."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The first input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return x

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The first input tensor.
            **kwargs (object): Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        import numpy as np

        return np.frexp(x)
