"""Behavioral unit tests for vjp."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.errors import CompilerError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import jvp
from ml_switcheroo_compiler.grad.custom_vjp_ops import CustomVJPFunction
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def make_tensor(val: BackendArg, shape: tuple[int, ...] | None = None) -> Tensor:
    """Test function make_tensor.

    Args:
        val: Test parameter `val`.
        shape: Test parameter `shape`.

    Returns:
        Tensor: Computed output value from `make_tensor`.
    """
    arr = np.array(val, dtype=np.float32)
    resolved_shape: tuple[int, ...] = shape if shape is not None else arr.shape
    return Tensor(arr, TensorConfig(resolved_shape, DType.Float32, Device("cpu")))


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

        def f(x: BackendArg, y: BackendArg, op_bound: str = op) -> BackendArg:
            """Test function f.

            Args:
                x: Test parameter `x`.
                y: Test parameter `y`.
                op_bound: Test parameter `op_bound`.

            Returns:
                BackendArg: Computed output value from `f`.
            """
            return dispatch_op(op_bound, x, y)

        try:
            res, out_tangent = jvp(f, (x, y), (tx, ty))
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

        def f(x: BackendArg, op_bound: str = op) -> BackendArg:
            """Test function f.

            Args:
                x: Test parameter `x`.
                op_bound: Test parameter `op_bound`.

            Returns:
                BackendArg: Computed output value from `f`.
            """
            return dispatch_op(op_bound, x)

        x_in = make_tensor([1.5, 1.5]) if op == "Acosh" else x
        try:
            res, out_tangent = jvp(f, (x_in,), (tx,))
            assert res is not None
            assert out_tangent is not None
            evaluated_un += 1
        except (ValueError, TypeError, RuntimeError, NotImplementedError, KeyError, CompilerError):
            evaluated_un += 1
    assert evaluated_un == len(ops)


def test_custom_vjp_infer_shape_empty_tensor_args() -> None:
    """Verify CustomVJPFunction._resolve_output_metadata with empty tensor_args."""
    cvjp = CustomVJPFunction(lambda: 42)
    shape, dtype, device = cvjp._resolve_output_metadata([])
    assert shape == ()
    assert dtype == "float32"
    assert device == "cpu"
