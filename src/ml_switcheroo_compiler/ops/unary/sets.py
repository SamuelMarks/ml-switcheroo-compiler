"""Core abstractions and logic definitions for sets.py."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Setdiff1d")
class Setdiff1d(OpDef):
    """Setdiff1d operator definition."""

    op_name = "Setdiff1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Setxor1d")
class Setxor1d(OpDef):
    """Setxor1d operator definition."""

    op_name = "Setxor1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("Union1d")
class Union1d(OpDef):
    """Union1d operator definition."""

    op_name = "Union1d"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueAll")
class UniqueAll(OpDef):
    """UniqueAll operator definition."""

    op_name = "UniqueAll"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueCounts")
class UniqueCounts(OpDef):
    """UniqueCounts operator definition."""

    op_name = "UniqueCounts"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueInverse")
class UniqueInverse(OpDef):
    """UniqueInverse operator definition."""

    op_name = "UniqueInverse"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueValues")
class UniqueValues(OpDef):
    """UniqueValues operator definition."""

    op_name = "UniqueValues"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns:
            object: Computed shape.
        """
        return args[0] if args else ()
