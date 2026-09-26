"""Eager operator execution dispatch for Pure Python backend."""

from __future__ import annotations

import math
from typing import Callable, Union

from ml_switcheroo_compiler.backends.pure_python.types import PurePythonTensor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError

Operand = Union[PurePythonTensor, int, float]
OpHandler = Callable[[tuple[Operand, ...]], Union[PurePythonTensor, float, int]]


def _to_tensor(val: Operand) -> PurePythonTensor:
    """Ensure operand is wrapped as a PurePythonTensor.

    Args:
        val (Operand): Scalar or tensor operand.

    Returns:
        PurePythonTensor: Wrapped tensor instance.
    """
    if isinstance(val, PurePythonTensor):
        return val
    return PurePythonTensor(val)


_OP_DISPATCH: dict[str, OpHandler] = {
    "add": lambda args: _to_tensor(args[0]) + _to_tensor(args[1]),
    "plus": lambda args: _to_tensor(args[0]) + _to_tensor(args[1]),
    "sub": lambda args: _to_tensor(args[0]) - _to_tensor(args[1]),
    "subtract": lambda args: _to_tensor(args[0]) - _to_tensor(args[1]),
    "minus": lambda args: _to_tensor(args[0]) - _to_tensor(args[1]),
    "mul": lambda args: _to_tensor(args[0]) * _to_tensor(args[1]),
    "multiply": lambda args: _to_tensor(args[0]) * _to_tensor(args[1]),
    "div": lambda args: _to_tensor(args[0]) / _to_tensor(args[1]),
    "divide": lambda args: _to_tensor(args[0]) / _to_tensor(args[1]),
    "truediv": lambda args: _to_tensor(args[0]) / _to_tensor(args[1]),
    "neg": lambda args: -_to_tensor(args[0]),
    "negative": lambda args: -_to_tensor(args[0]),
    "exp": lambda args: _to_tensor(args[0]).exp(),
    "log": lambda args: _to_tensor(args[0]).log(),
    "sqrt": lambda args: _to_tensor(args[0]).sqrt(),
    "sin": lambda args: _to_tensor(args[0]).sin(),
    "cos": lambda args: _to_tensor(args[0]).cos(),
    "tanh": lambda args: _to_tensor(args[0]).tanh(),
    "sum": lambda args: _to_tensor(args[0]).sum(),
    "reduce_sum": lambda args: _to_tensor(args[0]).sum(),
    "relu": lambda args: _to_tensor(args[0]).apply_elementwise(lambda x: max(0.0, x)),
    "sigmoid": lambda args: _to_tensor(args[0]).apply_elementwise(lambda x: 1.0 / (1.0 + math.exp(-x))),
}


def execute_op(
    cls_or_op: type | str,
    op_type_or_first: str | Operand | None = None,
    *args: Operand,
    **kwargs: int | float | str | bool,
) -> PurePythonTensor | float | int:
    """Execute operation eagerly using pure Python runtime primitives.

    Args:
        cls_or_op (type | str): Calling class context or operation name string.
        op_type_or_first (str | Operand | None): Operation name or first operand.
        *args (Operand): Positional input tensor or scalar parameters.
        **kwargs (int | float | str | bool): Operation configuration attributes.

    Returns:
        PurePythonTensor | float | int: Evaluated pure Python tensor or scalar.

    Raises:
        BackendNotSupportedError: When the operation cannot be executed in pure Python.
    """
    del kwargs
    op_type: str
    actual_args: tuple[Operand, ...]
    if isinstance(cls_or_op, type):
        op_type = str(op_type_or_first)
        actual_args = args
    else:
        op_type = str(cls_or_op)
        if op_type_or_first is not None and not isinstance(op_type_or_first, str):
            actual_args = (op_type_or_first,) + args
        else:
            actual_args = args

    op_norm: str = op_type.lower().replace("_", "")
    handler: OpHandler | None = _OP_DISPATCH.get(op_norm)
    if handler is not None:
        return handler(actual_args)

    raise BackendNotSupportedError(f"Operation '{op_type}' not supported in pure_python backend.")
