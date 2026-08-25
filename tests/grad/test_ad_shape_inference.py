import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.jvp_vjp import vjp
from ml_switcheroo_compiler.ops.dispatcher import dispatch_op


def make_tensor(val, shape=None):
    val = np.array(val, dtype=np.float32)
    if shape is None:
        shape = val.shape
    return Tensor(val, TensorConfig(shape, DType.Float32, Device("cpu")))


def test_vjp_shape_dtype_inference():
    def f(x, y):
        return dispatch_op("Multiply", x, y)

    x = make_tensor(np.ones((2, 3), dtype=np.float32))
    y = make_tensor(np.ones((2, 3), dtype=np.float32))

    out, vjp_fn = vjp(f, x, y)

    assert out.shape == (2, 3)
    assert out.dtype.value == "float32"

    cot = make_tensor(np.ones((2, 3), dtype=np.float32))
    grads = vjp_fn(cot)

    assert len(grads) == 2
    assert grads[0].shape == (2, 3)
    assert grads[1].shape == (2, 3)
