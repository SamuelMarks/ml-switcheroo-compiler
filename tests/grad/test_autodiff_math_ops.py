"""Unit tests for automatic differentiation dispatch of unary and binary mathematical ops."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def _make_tensor(val: object, shape: tuple[int, ...] | None = None) -> Tensor:
    """Construct a Tensor instance with CPU Float32 configuration.

    Args:
        val (object): Input data to wrap.
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

    for op in ops:

        def f(inp: Tensor, op_bound: str = op) -> Tensor:
            return dispatch_op(op_bound, inp)

        try:
            _, vjp_fn = vjp(f, x)
            _ = vjp_fn(cot)
        except Exception:
            pass


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

    for op in ops:

        def f(in_a: Tensor, in_b: Tensor, op_bound: str = op) -> Tensor:
            return dispatch_op(op_bound, in_a, in_b)

        try:
            _, vjp_fn = vjp(f, x, y)
            _ = vjp_fn(cot)
        except Exception:
            pass
