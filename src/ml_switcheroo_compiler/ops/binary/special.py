"""Defines special binary operations for the ml_switcheroo_compiler framework, including element-.

wise trigonometric, division, and comparison operations
"""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Atan2")
class Atan2(OpDef):
    """An operation class for computing the element-wise arc tangent of x/y."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Execute infer_shape.

        Args:
            *shapes (Any): Argument *shapes.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()


@register_op("Divmod")
class Divmod(OpDef):
    """An operation class for computing element-wise quotient and remainder."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Execute infer_shape.

        Args:
            *shapes (Any): Argument *shapes.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()


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


@register_op("Isclose")
class Isclose(OpDef):
    """An operation class for checking element-wise equality within a tolerance."""

    def infer_shape(self, *shapes: object, **kwargs: object) -> object:
        """Execute infer_shape.

        Args:
            *shapes (Any): Argument *shapes.
            **kwargs (Any): Argument **kwargs.

        Returns:
        Any: The result.
        """
        from ml_switcheroo_compiler.core.shape import broadcast_shapes as _bs

        """Infer the output shape of the operation.

        Args:
            *shapes: The input shapes.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed shape or evaluation result.
        """
        if all(isinstance(s, tuple) for s in shapes):
            return _bs(*shapes)
        return shapes[0] if shapes else ()
