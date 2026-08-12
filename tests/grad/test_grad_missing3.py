import numpy as np

from ml_switcheroo_compiler.core.config import ConfigContext
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import custom_vjp, hessian, jacfwd, jacrev, overwrite_with_gradient


def test_jacfwd():
    def f(x):
        return x * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = jacfwd(f)(x)
    assert res is not None
    # Analytical: df_i/dx_j = 2 * x_i if i==j else 0
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_jacrev():
    def f(x):
        return x * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = jacrev(f)(x)
    assert res is not None
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(res, expected, rtol=1e-5)


def test_hessian():
    from ml_switcheroo_compiler.core.config import ConfigContext

    def f(x):
        return x * x * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = hessian(f)(x)
    assert res is not None


def test_custom_vjp():
    @custom_vjp
    def f(x):
        return x * 2.0

    def fwd(x):
        return x * 2.0, x

    def bwd(res, g):
        return (g * 2.0,)

    f.defvjp(fwd, bwd)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        f(x)
        from ml_switcheroo_compiler.grad import grad

        g = grad(f)
        try:
            g(x)
        except Exception:
            pass


def test_overwrite_grad():
    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        overwrite_with_gradient(x, x)


def test_greater_jvp_vjp():
    from ml_switcheroo_compiler.grad import jvp, vjp

    def f(x, y):
        return x > y

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    y = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v1 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v2 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        jvp(f, (x, y), (v1, v2))
        vjp(f, x, y)
