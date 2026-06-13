"""Defines special binary operations for the ml_switcheroo_compiler framework, including element-.

wise trigonometric, division, and comparison operations
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Atan2")
class Atan2(OpDef):
    """An operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return np.broadcast_shapes(*shapes)
        return shapes[0] if shapes else ()

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args: The input arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return np.arctan2(args[0], args[1])


@register_op("Divmod")
class Divmod(OpDef):
    """An operation class for computing element-wise quotient and remainder."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return np.broadcast_shapes(*shapes)
        return shapes[0] if shapes else ()

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args: The input arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return np.divmod(args[0], args[1])


@register_op("Allclose")
class Allclose(OpDef):
    """An operation class for checking if two arrays are element-wise equal within a.

    tolerance
    """

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        return ()

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args: The input arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        rtol = kwargs.get("rtol", 1e-05)
        atol = kwargs.get("atol", 1e-08)
        equal_nan = kwargs.get("equal_nan", False)
        return np.allclose(args[0], args[1], rtol=rtol, atol=atol, equal_nan=equal_nan)


@register_op("Isclose")
class Isclose(OpDef):
    """An operation class for checking element-wise equality within a tolerance."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return np.broadcast_shapes(*shapes)
        return shapes[0] if shapes else ()

    def numpy_eval(self, *args: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            *args: The input arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        rtol = kwargs.get("rtol", 1e-05)
        atol = kwargs.get("atol", 1e-08)
        equal_nan = kwargs.get("equal_nan", False)
        return np.isclose(args[0], args[1], rtol=rtol, atol=atol, equal_nan=equal_nan)
