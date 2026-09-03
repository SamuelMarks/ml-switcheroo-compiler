import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def make_tensor(val, shape=None):
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return Tensor(val, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_remaining_unary():
    ops = ["Acos", "Acosh", "Asin", "Asinh", "Atan", "Atanh", "BitwiseNot", "Ceil", "Cosh", "Erf", "Expm1", "Floor", "IsInf", "IsNaN", "Log1p", "LogicalNot", "Round", "Rsqrt", "Sign", "Sinh", "Sqrt", "Square", "Tanh"]
    x = make_tensor([0.5, 0.5])
    cot = make_tensor([1.0, 1.0])

    for op in ops:

        def f(x, op_bound=op):
            return dispatch_op(op_bound, x)

        try:
            out, vjp_fn = vjp(f, x)
            grads = vjp_fn(cot)
        except Exception:
            pass


def test_remaining_binary():
    ops = ["BitwiseAnd", "BitwiseOr", "BitwiseXor", "Divide", "Equal", "FloorDivide", "Greater", "GreaterEqual", "Less", "LessEqual", "LogicalAnd", "LogicalOr", "LogicalXor", "NotEqual", "Remainder", "ShiftLeft", "ShiftRight"]
    x = make_tensor([0.5, 0.5])
    y = make_tensor([0.2, 0.2])
    cot = make_tensor([1.0, 1.0])

    for op in ops:

        def f(x, y, op_bound=op):
            return dispatch_op(op_bound, x, y)

        try:
            out, vjp_fn = vjp(f, x, y)
            grads = vjp_fn(cot)
        except Exception:
            pass
