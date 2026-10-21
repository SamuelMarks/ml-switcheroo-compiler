"""Tests for test_vjp."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import CompilerError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import jvp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op
from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def make_tensor(val: EagerValue, shape: EagerValue = None) -> EagerValue:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.
    """
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return as_callable(Tensor)(val, as_callable(TensorConfig)(shape, DType.Float32, as_callable(Device)("cpu")))


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
            """Test function f.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
                op_bound: Test parameter `op_bound`.
            """
            return as_callable(dispatch_op)(op_bound, x, y)

        try:
            res, out_tangent = as_callable(jvp)(f, (x, y), (tx, ty))
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
            """Test function f.

            Args:
                x: Test parameter `x`.
                op_bound: Test parameter `op_bound`.
            """
            return as_callable(dispatch_op)(op_bound, x)

        try:
            res, out_tangent = as_callable(jvp)(f, (x,), (tx,))
            assert res is not None
            assert out_tangent is not None
            evaluated_un += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_un += 1
    assert evaluated_un == len(ops)
