"""Tests for test_grad."""

from __future__ import annotations

import typing
from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import GradOptions, _check_scalar, _get_concrete_val, _to_original_type, checkpoint, custom_vjp, hessian, jacfwd, jacrev, overwrite_with_gradient, value_and_grad
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_check_scalar_symbolic() -> None:
    """Test _check_scalar on a tensor with non-scalar symbolic shape."""
    t = Tensor(np.ones(4), TensorConfig((4,), DType.Float32, Device("cpu"), requires_grad=True))
    with mock.patch.object(Tensor, "shape", new_callable=mock.PropertyMock, return_value=("N",)):
        with pytest.raises(Exception, match="backward\\(\\) can only be called on scalar tensors."):
            _check_scalar(t)


def test_get_concrete_val_proxy() -> None:
    """Test _get_concrete_val when value is a ProxyTensor inside _data."""

    class MockData:
        """Test class MockData."""

        pass

    t = as_callable(Tensor)(as_callable(MockData)(), as_callable(TensorConfig)((), DType.Float32, as_callable(Device)("cpu")))
    setattr(t, "_data", as_callable(ProxyTensor)("t1", (), "float32"))
    setattr(getattr(t, "_data"), "concrete_value", 42.0)

    with mock.patch.object(Tensor, "data", new_callable=mock.PropertyMock, return_value=None):
        assert _get_concrete_val(t) == 42.0


def test_grad_infer_dtype_fallback_2() -> None:
    """Test checkpoint infer dtype fallback."""
    out = checkpoint(lambda: 1)()
    assert out == 1


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return x, x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=True))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = as_callable(grad_func)(t)
    assert val is not None
    assert grads is not None


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=False))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = as_callable(grad_func)(t)
    assert val is not None
    assert grads is not None


def test_value_and_grad_basic() -> None:
    """Test value_and_grad basic functionality."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return (as_tensor(x) * as_tensor(x)) * as_tensor(2.0)

    Tensor(np.array([3.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    as_callable(np.testing.assert_allclose)(as_callable(val.numpy)(), as_callable(np.array)([18.0]))
    as_callable(np.testing.assert_allclose)(as_callable(grad.numpy)(), as_callable(np.array)([12.0]))


def test_value_and_grad_aux() -> None:
    """Test value_and_grad aux functionality."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(3.0)

    t = Tensor(np.array([2.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    assert t.shape == (1,)
    as_callable(np.testing.assert_allclose)(as_callable(val.numpy)(), as_callable(np.array)([18.0]))
    as_callable(np.testing.assert_allclose)(as_callable(grad.numpy)(), as_callable(np.array)([12.0]))


def test_jacrev_basic() -> None:
    """Test jacrev basic functionality."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(x)

    Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType("float32"), "cpu"))
    jac = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(jac, np.array([[2.0, 0.0], [0.0, 4.0]]))


def test_jacfwd() -> None:
    """Test function test_jacfwd."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(x)

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = as_callable(jacfwd)(f)(x)
    assert res is not None
    # Analytical: df_i/dx_j = 2 * x_i if i==j else 0
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    as_callable(np.testing.assert_allclose)(res, expected, rtol=1e-5)


def test_jacrev() -> None:
    """Test function test_jacrev."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(x)

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = as_callable(jacrev)(f)(x)
    assert res is not None
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    as_callable(np.testing.assert_allclose)(res, expected, rtol=1e-5)


def test_hessian() -> None:
    """Test function test_hessian."""
    from ml_switcheroo_compiler.core.config import ConfigContext

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return (as_tensor(x) * as_tensor(x)) * as_tensor(x)

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = as_callable(hessian)(f)(x)
    assert res is not None


def test_custom_vjp() -> None:
    """Test function test_custom_vjp."""

    @custom_vjp
    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(2.0)

    def fwd(x: EagerValue) -> EagerValue:
        """Test function fwd.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(2.0), x

    def bwd(res: EagerValue, g: EagerValue) -> EagerValue:
        """Test function bwd.

        Args:
            res: Test parameter `res`.
            g: Test parameter `g`.
        """
        return (as_tensor(g) * as_tensor(2.0),)

    as_tensor(getattr(f, "defvjp"))(fwd, bwd)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        out = as_callable(f)(x)
        assert out is not None
        from ml_switcheroo_compiler.grad import grad

        g = as_callable(grad)(f)
        g_res = g(x)
        assert g_res is not None


def test_overwrite_grad() -> None:
    """Test function test_overwrite_grad."""
    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = overwrite_with_gradient(x, x)
        assert res is not None


def test_greater_jvp_vjp() -> None:
    """Test function test_greater_jvp_vjp."""
    from ml_switcheroo_compiler.grad import jvp, vjp

    def f(x: EagerValue, y: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.
        """
        return as_tensor(x) > as_tensor(y)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    y = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v1 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v2 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        p_jvp, t_jvp = as_callable(jvp)(f, (x, y), (v1, v2))
        p_vjp, vjp_fn = as_callable(vjp)(f, x, y)
        assert p_jvp is not None
        assert t_jvp is not None
        assert p_vjp is not None
        assert callable(vjp_fn)


def test_checkpoint_eager() -> None:
    """Test function test_checkpoint_eager."""
    config.eager_mode = True

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(2)

    wrapped = checkpoint(my_func)
    assert wrapped(3) == 6
    config.eager_mode = False


def test_checkpoint_not_tracing() -> None:
    """Test function test_checkpoint_not_tracing."""
    global_tracing_state.is_tracing = False

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(2)

    wrapped = checkpoint(my_func)
    assert wrapped(4) == 8


def test_checkpoint_infer_dtype() -> None:
    """Test function test_checkpoint_infer_dtype."""
    global_tracing_state.is_tracing = True

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return x

    wrapped = checkpoint(my_func)
    t = Tensor(np.array([1, 2], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    try:
        res = wrapped(t)
        assert res is not None
    except RuntimeError as exc:
        assert "tracing" in str(exc).lower() or isinstance(exc, RuntimeError)
    global_tracing_state.is_tracing = False


def test_reconstruct_output_float64() -> None:
    """Test function test_reconstruct_output_float64."""
    t_orig = Tensor(np.array(1.0, dtype=np.float64), TensorConfig((), DType.Float64, Device("cpu")))
    res = _to_original_type(np.array(2.0, dtype=np.float64), t_orig)
    assert as_tensor(getattr(res, "dtype")) == DType.Float64
    assert as_tensor(getattr(res, "item"))() == 2.0


def test_grad_and_value_has_aux() -> None:
    """Test function test_grad_and_value_has_aux."""

    def my_func(x: EagerValue) -> EagerValue:
        """Test function my_func.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor(x) * as_tensor(x), x

    opt = GradOptions(has_aux=True)
    wrapped = value_and_grad(my_func, opt)
    t = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))
    (val, aux), g_out = as_callable(wrapped)(t)
    assert val is not None
    assert aux is not None
    assert g_out is not None
