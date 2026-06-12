"""Defines shape manipulation operations for the ML Switcheroo framework.

This module contains operator definitions (OpDefs) for reshaping, transposing, and
broadcasting tensors, along with their shape inference and NumPy evaluation logic
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Reshape")
class Reshape(OpDef):
    """An operator definition for reshaping a tensor to a new shape."""

    def infer_shape(self, x: object, newshape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            newshape (object): The newshape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return newshape

    def numpy_eval(self, x: object, newshape: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            newshape (object): The newshape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.reshape(x, newshape)


@register_op("Transpose")
class Transpose(OpDef):
    """An operator definition for transposing the dimensions of a tensor."""

    def infer_shape(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            axes (object): The axes parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        if isinstance(x, tuple) and axes is not None:
            return tuple(x[i] for i in axes)
        return None

    def numpy_eval(self, x: object, axes: object = None, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            axes (object): The axes parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.transpose(x, axes=axes)

    def _format_args(self, x: str, axes: object) -> str:
        """Evaluate format args.

        Args:
            x (str): Argument x
            axes (object): Argument axes
        """
        return f"{x}" if axes is None else f"{x}, {axes}"


@register_op("BroadcastTo")
class BroadcastTo(OpDef):
    """An operator definition for broadcasting a tensor to a new shape."""

    def infer_shape(self, x: object, shape: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return shape

    def numpy_eval(self, x: object, shape: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            shape (object): The shape parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.broadcast_to(x, shape)
