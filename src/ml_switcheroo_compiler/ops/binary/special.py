"""Defines special binary operations for the ml_switcheroo_compiler framework, including element-.

wise trigonometric, division, and comparison operations
"""

import numpy as np

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Atan2")
class Atan2(OpDef):
    """An operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.broadcast_shapes(x, y) if isinstance(x, tuple) and isinstance(y, tuple) else x

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.arctan2(x, y)


@register_op("Divmod")
class Divmod(OpDef):
    """An operation class for computing element-wise quotient and remainder."""

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.broadcast_shapes(x, y) if isinstance(x, tuple) and isinstance(y, tuple) else x

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.divmod(x, y)


@register_op("Allclose")
class Allclose(OpDef):
    """An operation class for checking if two arrays are element-wise equal within a.

    tolerance
    """

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return ()

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        rtol = kwargs.get("rtol", 1e-05)
        atol = kwargs.get("atol", 1e-08)
        equal_nan = kwargs.get("equal_nan", False)
        return np.allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)


@register_op("Isclose")
class Isclose(OpDef):
    """An operation class for checking element-wise equality within a tolerance."""

    def infer_shape(self, x: object, y: object, **kwargs: object) -> object:
        """Infer the output shape of the operation.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        return np.broadcast_shapes(x, y) if isinstance(x, tuple) and isinstance(y, tuple) else x

    def numpy_eval(self, x: object, y: object, **kwargs: object) -> object:
        """Evaluate the operation using NumPy.

        Args:
            x (object): The x parameter
            y (object): The y parameter
            **kwargs (object): Variable length argument list

        Returns:
            object: The resulting output
        """
        rtol = kwargs.get("rtol", 1e-05)
        atol = kwargs.get("atol", 1e-08)
        equal_nan = kwargs.get("equal_nan", False)
        return np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan)
