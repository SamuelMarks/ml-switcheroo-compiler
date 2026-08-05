"""Apply normalization basic operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("GroupMean")
class GroupMean(OpDef):
    """Compute the mean over groups."""

    op_name: str = "GroupMean"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
        object: Result.
        """
        return ()


@register_op("GroupVariance")
class GroupVariance(OpDef):
    """Compute the variance over groups."""

    op_name: str = "GroupVariance"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()


@register_op("GroupNorm")
class GroupNorm(OpDef):
    """Compute the group normalization."""

    op_name: str = "GroupNorm"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            object: Result.
        """
        return ()
