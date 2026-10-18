"""Apply normalization basic operations."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef, register_op


def _infer_group_stat_shape(
    args: tuple[tuple[int, ...] | int | None, ...],
    kwargs: dict[str, tuple[int, ...] | int | float | str | bool | None],
) -> tuple[int, ...]:
    """Infer output shape for GroupMean and GroupVariance operations.

    Args:
        args: Positional shape or tensor arguments.
        kwargs: Keyword arguments including num_groups and keepdims.

    Returns:
        tuple[int, ...]: Inferred group statistic shape.
    """
    if not args:
        return ()

    shape = args[0]
    if isinstance(shape, int):
        shape = (shape,)
    elif hasattr(shape, "shape"):
        shape = tuple(d for d in getattr(shape, "shape", getattr(shape, "shape_metadata", ())))

    if not isinstance(shape, tuple) or len(shape) <= 1:
        return tuple(d for d in shape) if isinstance(shape, tuple) else ()

    num_groups = kwargs.get("num_groups", shape[1])
    keepdims = kwargs.get("keepdims", True)

    import typing

    if not keepdims:
        return typing.cast(tuple[int, ...], (shape[0], num_groups))

    out_shape = [shape[0], num_groups] + [1] * (len(shape) - 2)
    return typing.cast(tuple[int, ...], tuple(out_shape))


@register_op("GroupMean")
class GroupMean(OpDef):
    """Compute the mean over groups."""

    op_name: str = "GroupMean"

    def infer_shape(
        self,
        *args: tuple[int, ...] | int | None,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[int, ...]: Computed result.
        """
        return _infer_group_stat_shape(args, kwargs)


@register_op("GroupVariance")
class GroupVariance(OpDef):
    """Compute the variance over groups."""

    op_name: str = "GroupVariance"

    def infer_shape(
        self,
        *args: tuple[int, ...] | int | None,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[int, ...]: Computed result.
        """
        return _infer_group_stat_shape(args, kwargs)


@register_op("GroupNorm")
class GroupNorm(OpDef):
    """Compute the group normalization."""

    op_name: str = "GroupNorm"

    def infer_shape(
        self,
        *args: tuple[int, ...] | int | None,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...]:
        """Infer shape.

        Args:
            *args: The args parameter.
            **kwargs: The kwargs parameter.

        Returns:
            tuple[int, ...]: Computed result.
        """
        if not args:
            return ()

        shape = args[0]
        if isinstance(shape, int):
            return (shape,)
        if hasattr(shape, "shape"):
            shape = tuple(d for d in getattr(shape, "shape", getattr(shape, "shape_metadata", ())))
        return tuple(d for d in getattr(shape, "__iter__")())
