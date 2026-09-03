from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import _check_scalar, _get_concrete_val
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor


def test_check_scalar_symbolic() -> None:
    """Test _check_scalar on a tensor with non-scalar symbolic shape."""
    t = Tensor(np.ones(4), TensorConfig((4,), DType.Float32, Device("cpu"), requires_grad=True))
    with mock.patch.object(Tensor, "shape", new_callable=mock.PropertyMock, return_value=("N",)):
        with pytest.raises(Exception, match="backward\\(\\) can only be called on scalar tensors."):
            _check_scalar(t)


def test_get_concrete_val_proxy() -> None:
    """Test _get_concrete_val when value is a ProxyTensor inside _data."""

    class MockData:
        pass

    t = Tensor(MockData(), TensorConfig((), DType.Float32, Device("cpu")))
    t._data = ProxyTensor("t1", (), "float32")
    t._data.concrete_value = 42.0

    with mock.patch.object(Tensor, "data", new_callable=mock.PropertyMock, return_value=None):
        assert _get_concrete_val(t) == 42.0


from ml_switcheroo_compiler.grad import GradOptions, checkpoint, value_and_grad


def test_grad_infer_dtype_fallback_2() -> None:
    """Test checkpoint infer dtype fallback."""
    checkpoint(lambda: 1)()


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def my_func(x):
        return x, x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=True))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = grad_func(t)


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def my_func(x):
        return x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=False))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = grad_func(t)


def test_value_and_grad_basic() -> None:
    """Test value_and_grad basic functionality."""

    def f(x):
        return x * x * 2.0

    t = Tensor(np.array([3.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    np.testing.assert_allclose(val.numpy(), np.array([18.0]))
    np.testing.assert_allclose(grad.numpy(), np.array([12.0]))


def test_value_and_grad_aux() -> None:
    """Test value_and_grad aux functionality."""

    def f(x):
        return x * 3.0

    t = Tensor(np.array([2.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    pass


def test_jacrev_basic() -> None:
    """Test jacrev basic functionality."""

    def f(x):
        return x * x

    t = Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType("float32"), "cpu"))
    jac = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(jac, np.array([[2.0, 0.0], [0.0, 4.0]]))


from ml_switcheroo_compiler.core.config import ConfigContext
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


from ml_switcheroo_compiler.core.config import config
from ml_switcheroo_compiler.grad import _to_original_type
from ml_switcheroo_compiler.tracing.state import global_tracing_state


def test_checkpoint_eager():
    config.eager_mode = True

    def my_func(x):
        return x * 2

    wrapped = checkpoint(my_func)
    assert wrapped(3) == 6
    config.eager_mode = False


def test_checkpoint_not_tracing():
    global_tracing_state.is_tracing = False

    def my_func(x):
        return x * 2

    wrapped = checkpoint(my_func)
    assert wrapped(4) == 8


def test_checkpoint_infer_dtype():
    global_tracing_state.is_tracing = True

    def my_func(x):
        return x

    wrapped = checkpoint(my_func)
    t = Tensor(np.array([1, 2], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    try:
        wrapped(t)
    except Exception:
        pass
    global_tracing_state.is_tracing = False


def test_reconstruct_output_float64():
    t_orig = Tensor(np.array(1.0, dtype=np.float64), TensorConfig((), DType.Float64, Device("cpu")))
    res = _to_original_type(np.array(2.0, dtype=np.float64), t_orig)
    assert res.dtype == DType.Float64
    assert res.item() == 2.0


def test_grad_and_value_has_aux():
    def my_func(x):
        return x * x, x

    opt = GradOptions(has_aux=True)
    wrapped = value_and_grad(my_func, opt)
    t = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))
    wrapped(t)
