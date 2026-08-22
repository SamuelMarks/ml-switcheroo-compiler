# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Core abstractions and logic definitions for sets.py."""

from typing import Any

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Setdiff1d")
class Setdiff1d(OpDef):
    """Setdiff1d operator definition."""

    op_name = "Setdiff1d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer the output shape for the infer_shape operation.

        Args:
        *args (object): Positional args.
        **kwargs (object): Keyword args.

        Returns:
            tuple[int, ...]: Result.
        """
        return args[0] if args else ()


@register_op("Setxor1d")
class Setxor1d(OpDef):
    """Setxor1d operator definition."""

    op_name = "Setxor1d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("Union1d")
class Union1d(OpDef):
    """Union1d operator definition."""

    op_name = "Union1d"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueAll")
class UniqueAll(OpDef):
    """UniqueAll operator definition."""

    op_name = "UniqueAll"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueCounts")
class UniqueCounts(OpDef):
    """UniqueCounts operator definition."""

    op_name = "UniqueCounts"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueInverse")
class UniqueInverse(OpDef):
    """UniqueInverse operator definition."""

    op_name = "UniqueInverse"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()


@register_op("UniqueValues")
class UniqueValues(OpDef):
    """UniqueValues operator definition."""

    op_name = "UniqueValues"

    def infer_shape(self, *args: Any, **kwargs: Any) -> Any:
        """Infer shape.

        Args:
            *args (object): Arguments.
            **kwargs (object): Keyword arguments.

        Returns: Any: Computed shape.
        """
        return args[0] if args else ()
