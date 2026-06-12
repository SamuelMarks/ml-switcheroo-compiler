"""Defines tensor creation operations for the ML Switcheroo framework.

This module contains operations that generate new tensors, such as zeros, ones, full,
and arange, along with their shape inference and NumPy evaluation implementations
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


class CreationOp(OpDef):
    """Base class for tensor creation operations.

    Provides common implementations for shape inference and NumPy evaluation
    for operations that create tensors of a specified shape (e.g., Zeros, Ones)
    """

    op_name: str = ""

    def infer_shape(self, shape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return shape

    def numpy_eval(self, shape: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return getattr(np, self.op_name.lower())(shape, **kwargs)


@register_op("Zeros")
class Zeros(CreationOp):
    """An operation that creates a tensor of a specified shape filled with zeros."""

    op_name = "Zeros"


@register_op("Ones")
class Ones(CreationOp):
    """An operation that creates a tensor of a specified shape filled with ones."""

    op_name = "Ones"


@register_op("Full")
class Full(CreationOp):
    """An operation that creates a tensor of a specified shape filled with a constant.

    value
    """

    op_name = "Full"

    def infer_shape(
        self,
        shape: object,
        fill_value: object,
        **kwargs: object,
    ) -> object:
        """Infer the output shape of the operation.

        Args:
            shape (object): The shape parameter
            fill_value (object): The fill_value parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return shape

    def numpy_eval(self, shape: object, fill_value: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            shape (object): The shape parameter
            fill_value (object): The fill_value parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.full(shape, fill_value, **kwargs)


@register_op("Arange")
class Arange(OpDef):
    """An operation that creates a 1-D tensor containing a sequence of evenly spaced.

    values

    within a given interval
    """

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *args (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return None  # Dynamic shape depending on values

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args (object): Variable length argument list
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.arange(*args, **kwargs)
