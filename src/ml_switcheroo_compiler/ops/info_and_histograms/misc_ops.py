# ruff: noqa: E402, D100, D103, D104, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, D101, D102, D107, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Misc operations."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Infeed")
class Infeed(OpDef):
    """Read from the infeed queue."""

    op_name = "Infeed"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return kwargs.get("shape", ())


@register_op("Vectorize")
class Vectorize(OpDef):
    """Generalized function class."""

    op_name = "Vectorize"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        return ()


@register_op("AxisIndex")
class AxisIndex(OpDef):
    """AxisIndex operation."""

    op_name = "AxisIndex"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Positional args.
            **kwargs (object): Keyword args.

        Returns: Any: Result.
        """
        if not args:
            return ()
        return getattr(args[0], "shape", ())


def infeed(*args: Any, **kwargs: Any) -> Any:
    """Read from the infeed queue.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Infeed", *args, **kwargs)


def vectorize(*args: Any, **kwargs: Any) -> Any:
    """Vectorize a python function.

    Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

    Returns: Any: Result.
    """
    from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

    return dispatch_op("Vectorize", *args, **kwargs)
