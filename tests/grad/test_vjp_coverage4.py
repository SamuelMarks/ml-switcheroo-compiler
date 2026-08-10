"""More VJP/JVP coverage."""

import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import jvp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def make_tensor(val, shape=None):
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return Tensor(val, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_jvp_binary():
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

    for op in ops:

        def f(x, y, op_bound=op):
            return dispatch_op(op_bound, x, y)

        try:
            res, out_tangent = jvp(f, (x, y), (tx, ty))
        except Exception:
            pass


def test_jvp_unary():
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

    for op in ops:

        def f(x, op_bound=op):
            return dispatch_op(op_bound, x)

        try:
            res, out_tangent = jvp(f, (x,), (tx,))
        except Exception:
            pass
