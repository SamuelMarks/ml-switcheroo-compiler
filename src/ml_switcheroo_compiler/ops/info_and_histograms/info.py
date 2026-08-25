# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Finfo")
class Finfo(OpDef):
    """Finfo operation."""

    op_name: object = "Finfo"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Iinfo")
class Iinfo(OpDef):
    """Iinfo operation."""

    op_name: object = "Iinfo"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("GetPrintoptions")
class GetPrintoptions(OpDef):
    """Get the current print options."""

    op_name: object = "GetPrintoptions"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Isscalar")
class Isscalar(OpDef):
    """Return True if the type of num is a scalar type."""

    op_name: object = "Isscalar"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("Iterable")
class Iterable(OpDef):
    """Check whether or not an object can be iterated over."""

    op_name: object = "Iterable"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("PromoteTypes")
class PromoteTypes(OpDef):
    """Return the data type with the smallest size and smallest scalar kind."""

    op_name: object = "PromoteTypes"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


@register_op("ResultType")
class ResultType(OpDef):
    """Return the type that results from applying the NumPy type promotion rules."""

    op_name: object = "ResultType"

    def infer_shape(self, *args: object, **kwargs: object) -> object:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return ()


def iinfo(*args: object, **kwargs: object) -> object:
    """Iinfo operation.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iinfo", *args, **kwargs)


def isscalar(*args: object, **kwargs: object) -> object:
    """Return True if the type of num is a scalar type.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Isscalar", *args, **kwargs)


def iterable(*args: object, **kwargs: object) -> object:
    """Check whether or not an object can be iterated over.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Iterable", *args, **kwargs)


def promote_types(*args: object, **kwargs: object) -> object:
    """Return the data type with the smallest size and smallest scalar kind.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("PromoteTypes", *args, **kwargs)


def result_type(*args: object, **kwargs: object) -> object:
    """Return the type that results from applying the NumPy type promotion rules.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("ResultType", *args, **kwargs)


def get_printoptions(*args: object, **kwargs: object) -> object:
    """Get the current print options.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns:
            tuple[int, ...]: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("GetPrintoptions", *args, **kwargs)
