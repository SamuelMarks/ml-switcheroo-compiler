"""Misc operations."""

from __future__ import annotations

from ml_switcheroo_compiler.core.shape import OpValue
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Finfo")
class Finfo(OpDef):
    """Finfo operation."""

    op_name = "Finfo"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Iinfo")
class Iinfo(OpDef):
    """Iinfo operation."""

    op_name = "Iinfo"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("GetPrintoptions")
class GetPrintoptions(OpDef):
    """Get the current print options."""

    op_name = "GetPrintoptions"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Isscalar")
class Isscalar(OpDef):
    """Return True if the type of num is a scalar type."""

    op_name = "Isscalar"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("Iterable")
class Iterable(OpDef):
    """Check whether or not an object can be iterated over."""

    op_name = "Iterable"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("PromoteTypes")
class PromoteTypes(OpDef):
    """Return the data type with the smallest size and smallest scalar kind."""

    op_name = "PromoteTypes"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


@register_op("ResultType")
class ResultType(OpDef):
    """Return the type that results from applying the NumPy type promotion rules."""

    op_name = "ResultType"

    def infer_shape(self, *args: OpValue, **kwargs: OpValue) -> OpValue:
        """Infer shape.

        Args:
            *args (OpValue): Positional args.
            **kwargs (OpValue): Keyword args.

        Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
        """
        return ()


def iinfo(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Iinfo operation.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iinfo", *args, **kwargs)


def isscalar(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return True if the type of num is a scalar type.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isscalar", *args, **kwargs)


def iterable(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Check whether or not an object can be iterated over.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iterable", *args, **kwargs)


def promote_types(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return the data type with the smallest size and smallest scalar kind.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PromoteTypes", *args, **kwargs)


def result_type(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Return the type that results from applying the NumPy type promotion rules.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ResultType", *args, **kwargs)


def get_printoptions(*args: OpValue, **kwargs: OpValue) -> OpValue:
    """Get the current print options.

    Args:
        *args (OpValue): Positional args.
        **kwargs (OpValue): Keyword args.

    Returns:
            tuple[int, ...]: Computed `tuple[int, ...]` output.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("GetPrintoptions", *args, **kwargs)
