"""Normalization basic operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("GroupMean")
class GroupMean(OpDef):
    """Computes the mean over groups."""

    op_name: str = "GroupMean"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("GroupVariance")
class GroupVariance(OpDef):
    """Computes the variance over groups."""

    op_name: str = "GroupVariance"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()


@register_op("GroupNorm")
class GroupNorm(OpDef):
    """Computes the group normalization."""

    op_name: str = "GroupNorm"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape."""
        return ()
