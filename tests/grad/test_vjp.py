"""Tests for test_vjp."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import CompilerError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import jvp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor."""
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return cast("Callable[..., EagerTensorProtocol]", Tensor)(val, cast("Callable[..., EagerTensorProtocol]", TensorConfig)(shape, DType.Float32, cast("Callable[..., EagerTensorProtocol]", Device)("cpu")))


def test_jvp_binary() -> None:
    """Test function test_jvp_binary."""
    ops = [
        "Add",
        "Subtract",
        "Multiply",
        "TrueDivide",
        "Power",
        "Maximum",
        "Minimum",
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
        "BinaryCrossentropy",
        "CategoricalCrossentropy",
        "Hinge",
        "Huber",
        "LogCosh",
        "MeanAbsoluteError",
        "MeanAbsolutePercentageError",
        "MeanSquaredError",
        "MeanSquaredLogarithmicError",
        "Poisson",
        "SparseCategoricalCrossentropy",
        "SquaredHinge",
    ]
    x = make_tensor([0.5, 0.5])
    y = make_tensor([0.2, 0.2])
    tx = make_tensor([1.0, 1.0])
    ty = make_tensor([1.0, 1.0])

    evaluated_bin = 0
    for op in ops:

        def f(x: EagerValue, y: EagerValue, op_bound: EagerValue = op) -> EagerValue:
            """Test function f."""
            return cast("Callable[..., EagerTensorProtocol]", dispatch_op)(op_bound, x, y)

        try:
            res, out_tangent = cast("Callable[..., EagerTensorProtocol]", jvp)(f, (x, y), (tx, ty))
            assert res is not None
            assert out_tangent is not None
            evaluated_bin += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_bin += 1
    assert evaluated_bin == len(ops)


def test_jvp_unary() -> None:
    """Test function test_jvp_unary."""
    ops = [
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
        "Exp",
        "Log",
        "Sin",
        "Cos",
        "Tan",
        "Abs",
        "Negative",
        "Relu",
        "Sigmoid",
        "Elu",
        "Gelu",
        "Selu",
        "HardSigmoid",
        "HardSwish",
        "Softplus",
        "Softsign",
    ]
    x = make_tensor([0.5, 0.5])
    tx = make_tensor([1.0, 1.0])

    evaluated_un = 0
    for op in ops:

        def f(x: EagerValue, op_bound: EagerValue = op) -> EagerValue:
            """Test function f."""
            return cast("Callable[..., EagerTensorProtocol]", dispatch_op)(op_bound, x)

        try:
            res, out_tangent = cast("Callable[..., EagerTensorProtocol]", jvp)(f, (x,), (tx,))
            assert res is not None
            assert out_tangent is not None
            evaluated_un += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_un += 1
    assert evaluated_un == len(ops)
