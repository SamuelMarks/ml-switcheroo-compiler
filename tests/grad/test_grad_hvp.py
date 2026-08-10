import numpy as np

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import check_numerical_grads, hvp


def test_hvp_robustness():
    def f(x):
        return x * x * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    val, out_tan = hvp(f, x, v)
    assert val.item() == 8.0
    assert out_tan.item() == 12.0  # 6 * x * v = 6 * 2 * 1 = 12


def test_hessian():
    def f(x):
        return x * x * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))

    from ml_switcheroo_compiler.grad import hessian

    res = hessian(f)(x)
    assert res is not None


def test_check_numerical_grads():
    def f(x):
        return x * x

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    check_numerical_grads(f, (x,))
