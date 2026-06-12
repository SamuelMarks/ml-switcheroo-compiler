"""Defines special unary operations for the ML Switcheroo framework, including Cast,.

Bitcast, and Frexp
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


import numpy as np

from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Cast")
class Cast(OpDef):
    """An operation that casts an input array to a specified data type."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return x

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        dtype = kwargs.get("dtype")
        if isinstance(dtype, DType):
            dtype = dtype.value
        return np.array(x).astype(dtype)


@register_op("Bitcast")
class Bitcast(Cast):
    """An operation that bitcasts an input array to a specified data type without copying."""

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        dtype = kwargs.get("dtype")
        if isinstance(dtype, DType):
            dtype = dtype.value
        return np.array(x).view(dtype)


@register_op("Frexp")
class Frexp(OpDef):
    """An operation that decomposes a floating-point array into mantissa and exponent."""

    def infer_shape(self, x: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return x

    def numpy_eval(self, x: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.frexp(x)
