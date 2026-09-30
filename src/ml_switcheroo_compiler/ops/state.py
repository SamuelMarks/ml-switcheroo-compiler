# ruff: noqa: E402, F401, E501, C901, PLR0911, PLR0912, F841, PLR0917, F811, B018, E701, E722, F403, E711, E712, PLR0913, PLR0915
"""Define stateful operations for reading and writing variables within the ML Switcheroo framework."""

from __future__ import annotations

from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op("ReadVariable")
class ReadVariable(OpDef):
    """Provide an operation definition for reading the value of a stateful variable within the computational graph."""

    def infer_shape(
        self,
        *args: tuple[int, ...] | int | None,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...] | str:
        """Infer the output shape of the operation.

        Args:
            *args (tuple[int, ...] | int | None): Additional positional arguments.
            **kwargs (tuple[int, ...] | int | float | str | bool | None): Additional keyword arguments.

        Returns:
            tuple[int, ...] | str: The computed shape or evaluation result.
        """
        shape_val = kwargs.get("shape", ())
        if isinstance(shape_val, tuple):
            return shape_val
        if isinstance(shape_val, str):
            return shape_val
        return ()


@register_op("AssignVariable")
class AssignVariable(OpDef):
    """Provide an operation definition for assigning a new value to a stateful variable within the computational graph."""

    def infer_shape(
        self,
        x: tuple[int, ...] | str,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...] | str:
        """Infer the output shape of the operation.

        Args:
            x (tuple[int, ...] | str): The first input tensor shape or symbol.
            **kwargs (tuple[int, ...] | int | float | str | bool | None): Additional keyword arguments.

        Returns:
            tuple[int, ...] | str: The computed shape or evaluation result.
        """
        return x
