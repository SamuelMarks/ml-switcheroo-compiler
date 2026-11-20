"""Apply normalization basic operations."""

from __future__ import annotations

import typing

from ml_switcheroo_compiler.core.shape import ShapeProtocol, extract_symbolic_shape
from ml_switcheroo_compiler.ir.shape_system import SymInt, SymNode
from ml_switcheroo_compiler.ops.base import OpDef, register_op

_GroupDim = typing.Union[int, str, SymNode, SymInt]  # noqa: UP007
_GroupShapeArg = typing.Union[tuple[_GroupDim, ...], _GroupDim, ShapeProtocol, None]  # noqa: UP007
_GroupKwarg = typing.Union[tuple[_GroupDim, ...], _GroupDim, float, bool, None]  # noqa: UP007


def _to_group_dim(val: _GroupKwarg) -> _GroupDim:
    """Normalize a single group dimension value while preserving symbolic dimensions.

    Args:
        val (_GroupKwarg): Candidate integer or symbolic dimension value.

    Returns:
        _GroupDim: Preserved integer or symbolic dimension.
    """
    if isinstance(val, (SymNode, SymInt, str)):
        return val
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, float):
        return int(val)
    return 1


def _extract_group_input_shape(first: _GroupShapeArg) -> tuple[_GroupDim, ...] | None:
    """Extract an integer or symbolic shape tuple from `first`.

    Args:
        first (_GroupShapeArg): Input shape tuple, dimension, or shape-bearing object.

    Returns:
        tuple[_GroupDim, ...] | None: Extracted shape tuple or `None`.
    """
    if isinstance(first, (int, str, SymNode, SymInt)) and not isinstance(first, bool):
        return (first,)
    if isinstance(first, tuple):
        return tuple(_to_group_dim(d) for d in first)
    if isinstance(first, ShapeProtocol):
        return tuple(_to_group_dim(d) for d in extract_symbolic_shape(first))
    return None


def _infer_group_stat_shape(
    args: tuple[_GroupShapeArg, ...],
    kwargs: dict[str, _GroupKwarg],
) -> tuple[_GroupDim, ...]:
    """Infer output shape for GroupMean and GroupVariance operations.

    Args:
        args (tuple[_GroupShapeArg, ...]): Positional shape or tensor arguments.
        kwargs (dict[str, _GroupKwarg]): Keyword arguments including `num_groups` and `keepdims`.

    Returns:
        tuple[_GroupDim, ...]: Inferred group statistic shape.
    """
    if not args:
        return ()

    shape = _extract_group_input_shape(args[0])
    if not isinstance(shape, tuple) or len(shape) <= 1:
        return shape if isinstance(shape, tuple) else ()

    num_groups = _to_group_dim(kwargs.get("num_groups", shape[1]))
    keepdims = kwargs.get("keepdims", True)

    if not keepdims:
        return (shape[0], num_groups)

    out_shape: list[_GroupDim] = [shape[0], num_groups] + [1] * (len(shape) - 2)
    return tuple(out_shape)


@register_op("GroupMean")
class GroupMean(OpDef):
    """Compute the mean over groups."""

    op_name: str = "GroupMean"

    def infer_shape(self, *args: _GroupShapeArg, **kwargs: _GroupKwarg) -> tuple[_GroupDim, ...]:
        """Infer output shape of `GroupMean`.

        Args:
            *args (_GroupShapeArg): Input shapes or tensor metadata.
            **kwargs (_GroupKwarg): Keyword options (`num_groups`, `keepdims`).

        Returns:
            tuple[_GroupDim, ...]: Inferred statistic shape.
        """
        return _infer_group_stat_shape(args, kwargs)


@register_op("GroupVariance")
class GroupVariance(OpDef):
    """Compute the variance over groups."""

    op_name: str = "GroupVariance"

    def infer_shape(self, *args: _GroupShapeArg, **kwargs: _GroupKwarg) -> tuple[_GroupDim, ...]:
        """Infer output shape of `GroupVariance`.

        Args:
            *args (_GroupShapeArg): Input shapes or tensor metadata.
            **kwargs (_GroupKwarg): Keyword options (`num_groups`, `keepdims`).

        Returns:
            tuple[_GroupDim, ...]: Inferred statistic shape.
        """
        return _infer_group_stat_shape(args, kwargs)


@register_op("GroupNorm")
class GroupNorm(OpDef):
    """Compute the group normalization."""

    op_name: str = "GroupNorm"

    def infer_shape(self, *args: _GroupShapeArg, **kwargs: _GroupKwarg) -> tuple[_GroupDim, ...]:
        """Infer output shape of `GroupNorm`.

        Args:
            *args (_GroupShapeArg): Input shapes or tensor metadata.
            **kwargs (_GroupKwarg): Unused keyword options.

        Returns:
            tuple[_GroupDim, ...]: Inferred normalized tensor shape.
        """
        del kwargs
        if not args:
            return ()
        shape = _extract_group_input_shape(args[0])
        return shape if shape is not None else ()
