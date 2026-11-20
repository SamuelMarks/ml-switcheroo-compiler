"""Registered stateful variable operations (State Layer 4).

Defines canonical ``OpDef`` schemas for ``ReadVariable`` and ``AssignVariable``
using operation identifiers from ``ml_switcheroo_compiler.ir.state``.
"""

from __future__ import annotations

from ml_switcheroo_compiler.ir.state import (
    ASSIGN_VARIABLE_OP as ASSIGN_VARIABLE_OP,
)
from ml_switcheroo_compiler.ir.state import (
    READ_VARIABLE_OP as READ_VARIABLE_OP,
)
from ml_switcheroo_compiler.ops.base import OpDef, register_op


@register_op(READ_VARIABLE_OP)
class ReadVariable(OpDef):
    """Operation definition for reading the current value of a named state variable."""

    def infer_shape(
        self,
        *args: tuple[int, ...] | int | None,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...] | str:
        """Infer the output shape of ``ReadVariable`` from its ``shape`` attribute.

        Args:
            *args (tuple[int, ...] | int | None): Unused positional arguments.
            **kwargs (tuple[int, ...] | int | float | str | bool | None): Node attributes containing ``shape``.

        Returns:
            tuple[int, ...] | str: Declared variable shape tuple or symbolic identifier.
        """
        del args
        shape_val = kwargs.get("shape", ())
        if isinstance(shape_val, tuple):
            return shape_val
        if isinstance(shape_val, str):
            return shape_val
        return ()


@register_op(ASSIGN_VARIABLE_OP)
class AssignVariable(OpDef):
    """Operation definition for assigning a new value to a named state variable."""

    def infer_shape(
        self,
        x: tuple[int, ...] | str,
        **kwargs: tuple[int, ...] | int | float | str | bool | None,
    ) -> tuple[int, ...] | str:
        """Infer the output shape of ``AssignVariable`` as the assigned value's shape.

        Args:
            x (tuple[int, ...] | str): Assigned value tensor shape or symbolic identifier.
            **kwargs (tuple[int, ...] | int | float | str | bool | None): Unused keyword attributes.

        Returns:
            tuple[int, ...] | str: Output shape matching ``x``.
        """
        del kwargs
        return x
