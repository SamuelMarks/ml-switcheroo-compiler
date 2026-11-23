"""Provide `InterpreterValue` and `Environment` for variable state and tensor memory mappings during evaluation."""

from __future__ import annotations

import typing
from collections.abc import Mapping

from ml_switcheroo_compiler.backends.eager_registry import ShapeProtocol
from ml_switcheroo_compiler.core.shape import OpValue, Scalar

InterpreterValue = typing.Union[  # noqa: UP007
    ShapeProtocol,
    Scalar,
    str,
    None,
    list["InterpreterValue"],
    tuple["InterpreterValue", ...],
]
ValueType = InterpreterValue


def _is_eval_shape_value(val: OpValue | InterpreterValue) -> typing.TypeGuard[ShapeProtocol]:
    """Check whether `val` satisfies `ShapeProtocol` including mock tensors.

    Args:
        val (OpValue | InterpreterValue): Candidate runtime value.

    Returns:
        typing.TypeGuard[ShapeProtocol]: `True` when `val` exposes `.shape`.
    """
    from ml_switcheroo_compiler.core._mock_protocol import is_mock_with_attrs

    found: bool = isinstance(val, ShapeProtocol)
    return found or is_mock_with_attrs(val, "shape")


def as_eval_value(val: OpValue | InterpreterValue) -> InterpreterValue:
    """Validate and narrow a candidate runtime value to `InterpreterValue`.

    Args:
        val (OpValue | InterpreterValue): Candidate array, scalar, or nested tuple/list value.

    Returns:
        InterpreterValue: Validated runtime evaluation value.

    Raises:
        TypeError: If `val` is not a valid runtime tensor, scalar, or sequence value.
    """
    if val is None or isinstance(val, (int, float, bool, complex, str)) or _is_eval_shape_value(val):
        return val
    if isinstance(val, tuple):
        return tuple(as_eval_value(item) for item in val)
    if isinstance(val, list):
        return [as_eval_value(item) for item in val]
    msg = f"Unsupported non-tensor evaluation value of type {type(val).__name__}"
    raise TypeError(msg)


class Environment:
    """Manages variable state, tensor memory mappings, and inputs during interpretation.

    This class acts as a symbol table or memory store, mapping variable names
    to their corresponding values or tensors during the evaluation of an expression
    or execution of a graph.

    Attributes:
        memory (dict[str, InterpreterValue]): The internal storage mapping variable names to their values.
    """

    memory: dict[str, InterpreterValue]

    def __init__(self, inputs: Mapping[str, InterpreterValue] | None = None) -> None:
        """Initialize the environment.

        Args:
            inputs (Mapping[str, InterpreterValue] | None): Initial variable mapping dictionary.
        """
        self.memory = dict(inputs) if inputs is not None else {}

    def get(self, name: str) -> InterpreterValue:
        """Retrieve the value associated with the given node or variable name.

        Args:
            name (str): The unique identifier for the tensor or variable to retrieve.

        Returns:
            InterpreterValue: The concrete tensor, scalar, or value associated with the name.

        Raises:
            ValueError: If the requested name does not exist in the environment's memory.
        """
        if name not in self.memory:
            msg = f"Missing input value for node '{name}'"
            raise ValueError(msg)
        return self.memory[name]

    def set(self, name: str, value: InterpreterValue) -> None:
        """Store or update a value in the environment for a specific node or variable.

        Args:
            name (str): The unique identifier where the value should be stored.
            value (InterpreterValue): The concrete tensor, scalar, or tuple to store.
        """
        self.memory[name] = value

    def __contains__(self, name: str) -> bool:
        """Check if a specific node or variable name exists within the environment.

        Args:
            name (str): The unique identifier to check for in the memory store.

        Returns:
            bool: True if the name is present in the environment's memory, False otherwise.
        """
        return name in self.memory
