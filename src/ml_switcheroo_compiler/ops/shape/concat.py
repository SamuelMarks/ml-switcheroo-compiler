# pylint: disable=duplicate-code

"""Defines shape manipulation operations for the ML Switcheroo framework."""

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("Concatenate")
class Concatenate(OpDef):
    """Concatenate operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Concatenate."""
        return ()


@register_op("Stack")
class Stack(OpDef):
    """Stack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Stack."""
        return ()


@register_op("Split")
class Split(OpDef):
    """Split operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Split."""
        return ()


@register_op("Hsplit")
class Hsplit(OpDef):
    """Hsplit operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Hsplit."""
        return ()


@register_op("Vsplit")
class Vsplit(OpDef):
    """Vsplit operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Vsplit."""
        return ()


@register_op("Dsplit")
class Dsplit(OpDef):
    """Dsplit operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Dsplit."""
        return ()


@register_op("Hstack")
class Hstack(OpDef):
    """Hstack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Hstack."""
        return ()


@register_op("Vstack")
class Vstack(OpDef):
    """Vstack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Vstack."""
        return ()


@register_op("Dstack")
class Dstack(OpDef):
    """Dstack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for Dstack."""
        return ()


@register_op("ColumnStack")
class ColumnStack(OpDef):
    """ColumnStack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for ColumnStack."""
        return ()


@register_op("RowStack")
class RowStack(OpDef):
    """RowStack operator definition."""

    def infer_shape(self, *args: object, **kwargs: object) -> tuple[int, ...]:
        """Infer shape for RowStack."""
        return ()


@register_op("Argwhere")
class Argwhere(OpDef):
    """Find the indices of array elements that are non-zero, grouped by element."""

    op_name = "Argwhere"
    np_op_name = "argwhere"

    def infer_shape(self, a: object, **kwargs: object) -> object:
        """Infer the output shape."""
        return (None, len(a) if isinstance(a, tuple) else None)
