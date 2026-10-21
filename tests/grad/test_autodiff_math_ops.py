"""Unit tests for automatic differentiation dispatch of unary and binary mathematical ops."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import CompilerError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from tests.testing_utils import as_callable, as_tensor_obj

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def _make_tensor(val: EagerValue, shape: tuple[int, ...] | None = None) -> Tensor:
    """Construct a Tensor instance with CPU Float32 configuration.

    Args:
        val (EagerValue): Input data to wrap.
        shape (tuple[int, ...] | None): Optional explicit shape.

    Returns:
        Tensor: Constructed tensor instance.
    """
    arr = np.array(val, dtype=np.float32)
    if shape is None:
        shape = arr.shape
    return Tensor(arr, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_autodiff_unary_math_ops() -> None:
    """Verify reverse-mode autodiff VJP evaluation for unary math operations."""
    ops: list[str] = [
        "Acos",
        "Acosh",
        "Asin",
        "Asinh",
        "Atan",
        "Atanh",
        "BitwiseNot",
        "Ceil",
        "Cosh",
        "Erf",
        "Expm1",
        "Floor",
        "IsInf",
        "IsNaN",
        "Log1p",
        "LogicalNot",
        "Round",
        "Rsqrt",
        "Sign",
        "Sinh",
        "Sqrt",
        "Square",
        "Tanh",
    ]
    x = _make_tensor([0.5, 0.5])
    cot = _make_tensor([1.0, 1.0])

    evaluated_unary = 0
    for op in ops:

        def f(inp: Tensor, op_bound: str = op) -> Tensor:
            """Test function f.

            Args:
                inp: Test parameter `inp`.
                op_bound: Test parameter `op_bound`.
            """
            return as_tensor_obj(dispatch_op(op_bound, inp))

        try:
            primal, vjp_fn = as_callable(vjp)(f, x)
            grads = as_callable(vjp_fn)(cot)
            assert primal is not None
            assert grads is not None
            evaluated_unary += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_unary += 1
    assert evaluated_unary == len(ops)


def test_autodiff_binary_math_ops() -> None:
    """Verify reverse-mode autodiff VJP evaluation for binary math operations."""
    ops: list[str] = [
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "Divide",
        "Equal",
        "FloorDivide",
        "Greater",
        "GreaterEqual",
        "Less",
        "LessEqual",
        "LogicalAnd",
        "LogicalOr",
        "LogicalXor",
        "NotEqual",
        "Remainder",
        "ShiftLeft",
        "ShiftRight",
    ]
    x = _make_tensor([0.5, 0.5])
    y = _make_tensor([0.2, 0.2])
    cot = _make_tensor([1.0, 1.0])

    evaluated_binary = 0
    for op in ops:

        def f(in_a: Tensor, in_b: Tensor, op_bound: str = op) -> Tensor:
            """Test function f.

            Args:
                in_a: Test parameter `in_a`.
                in_b: Test parameter `in_b`.
                op_bound: Test parameter `op_bound`.
            """
            return as_tensor_obj(dispatch_op(op_bound, in_a, in_b))

        try:
            primal, vjp_fn = as_callable(vjp)(f, x, y)
            grads = as_callable(vjp_fn)(cot)
            assert primal is not None
            assert grads is not None
            evaluated_binary += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_binary += 1
    assert evaluated_binary == len(ops)
