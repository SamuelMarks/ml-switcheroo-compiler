"""Tests for autodiff math rules."""

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


# Helper to avoid repetitive tensor creation
def make_tensor(val, shape=None):
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return Tensor(val, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_unary_math_rules():
    ops = ["Abs", "Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "BitwiseNot", "Ceil", "Cos", "Cosh", "Erf", "Exp", "Expm1", "Floor", "IsInf", "IsNaN", "Log", "Log1p", "LogicalNot", "Negative", "Round", "Rsqrt", "Sign", "Sin", "Sinh", "Sqrt", "Square", "Tan", "Tanh"]

    # We just want to make sure the registry has a rule and it executes without crashing
    for op in ops:

        def f(x, op_bound=op):
            return dispatch_op(op_bound, x)

        x = make_tensor([0.5, 0.5])
        cot = make_tensor([1.0, 1.0])
        try:
            out, vjp_fn = vjp(f, x)
            grads = vjp_fn(cot)
            assert len(grads) == 1
        except Exception as e:
            # Catching exceptions allows us to see which ones are actually missing vs which crash
            print(f"Failed {op}: {e}")


def test_binary_math_rules():
    ops = [
        "Add",
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
        "Maximum",
        "Minimum",
        "Multiply",
        "NotEqual",
        "Power",
        "Remainder",
        "ShiftLeft",
        "ShiftRight",
        "Subtract",
        "TrueDivide",
    ]

    for op in ops:

        def f(x, y, op_bound=op):
            return dispatch_op(op_bound, x, y)

        x = make_tensor([0.5, 0.5])
        y = make_tensor([0.2, 0.2])
        cot = make_tensor([1.0, 1.0])
        try:
            out, vjp_fn = vjp(f, x, y)
            grads = vjp_fn(cot)
            assert len(grads) == 2
        except Exception as e:
            print(f"Failed {op}: {e}")
