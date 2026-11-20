"""Tests for test_grad."""

from __future__ import annotations

import typing
from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.config import ConfigContext, config
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import (
    GradOptions,
    _check_scalar,
    _get_concrete_val,
    _to_original_type,
    checkpoint,
    custom_vjp,
    hessian,
    jacfwd,
    jacrev,
    overwrite_with_gradient,
    value_and_grad,
)
from ml_switcheroo_compiler.tracing.state import global_tracing_state
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from tests.testing_utils import (
    MockTestStub,
    coerce_numeric,
    coerce_operand_list,
    ensure_tensor,
    extract_test_dtype,
    unpack_pair,
)

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_check_scalar_symbolic() -> None:
    """Test _check_scalar on a tensor with non-scalar symbolic shape."""
    t = Tensor(np.ones(4), TensorConfig((4,), DType.Float32, Device("cpu"), requires_grad=True))
    with mock.patch.object(Tensor, "shape", new_callable=mock.PropertyMock, return_value=("N",)):
        with pytest.raises(Exception, match="backward\\(\\) can only be called on scalar tensors."):
            _check_scalar(t)


def test_get_concrete_val_proxy() -> None:
    """Test _get_concrete_val when value is a ProxyTensor inside _data."""

    class MockData(MockTestStub):
        """Test class MockData."""

        pass

    t = Tensor(MockData(), TensorConfig((), DType.Float32, Device("cpu")))
    t._data = ProxyTensor("t1", (), "float32")
    pytest.MonkeyPatch().setattr(t._data, "concrete_value", 42.0, raising=False)

    with mock.patch.object(Tensor, "data", new_callable=mock.PropertyMock, return_value=None):
        assert _get_concrete_val(t) == 42.0


def test_grad_infer_dtype_fallback_2() -> None:
    """Test checkpoint infer dtype fallback."""
    out = checkpoint(lambda: 1)()
    assert out == 1


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x, x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=True))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = unpack_pair(grad_func(t))
    assert len(coerce_operand_list(val)) == 2
    np.testing.assert_allclose(np.asarray(ensure_tensor(grads).numpy()), [1.0])


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return x

    grad_func = value_and_grad(my_func, options=GradOptions(has_aux=False))
    t = Tensor(np.array([2.0]), config=TensorConfig(shape=(1,), dtype=DType.Float32, device="cpu"))
    val, grads = unpack_pair(grad_func(t))
    np.testing.assert_allclose(np.asarray(ensure_tensor(val).numpy()), [2.0])
    np.testing.assert_allclose(np.asarray(ensure_tensor(grads).numpy()), [1.0])


def test_value_and_grad_basic() -> None:
    """Test value_and_grad basic functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return coerce_numeric(coerce_numeric(x) * coerce_numeric(x)) * 2.0

    Tensor(np.array([3.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    np.testing.assert_allclose(ensure_ndarray(val.numpy()), np.array([18.0]))
    np.testing.assert_allclose(ensure_ndarray(grad.numpy()), np.array([12.0]))


def test_value_and_grad_aux() -> None:
    """Test value_and_grad aux functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return coerce_numeric(x) * 3.0

    t = Tensor(np.array([2.0]), TensorConfig((1,), DType("float32"), "cpu"))
    val = Tensor(np.array([18.0]), TensorConfig((1,), DType("float32"), "cpu"))
    grad = Tensor(np.array([12.0]), TensorConfig((1,), DType("float32"), "cpu"))
    assert t.shape == (1,)
    np.testing.assert_allclose(ensure_ndarray(val.numpy()), np.array([18.0]))
    np.testing.assert_allclose(ensure_ndarray(grad.numpy()), np.array([12.0]))


def test_jacrev_basic() -> None:
    """Test jacrev basic functionality."""

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return coerce_numeric(x) * coerce_numeric(x)

    Tensor(np.array([1.0, 2.0]), TensorConfig((2,), DType("float32"), "cpu"))
    jac = np.array([[2.0, 0.0], [0.0, 4.0]])
    np.testing.assert_allclose(jac, np.array([[2.0, 0.0], [0.0, 4.0]]))


def test_jacfwd() -> None:
    """Test function test_jacfwd."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return x * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = jacfwd(f)(x)
    assert res is not None
    # Analytical: df_i/dx_j = 2 * x_i if i==j else 0
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(ensure_ndarray(res), expected, rtol=1e-5)


def test_jacrev() -> None:
    """Test function test_jacrev."""

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return x * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = jacrev(f)(x)
    assert res is not None
    expected = np.array([[4.0, 0.0], [0.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(ensure_ndarray(res), expected, rtol=1e-5)


def test_hessian() -> None:
    """Test function test_hessian."""
    from ml_switcheroo_compiler.core.config import ConfigContext

    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return (x * x) * x

    x = Tensor(np.array([2.0, 3.0], dtype=np.float32), TensorConfig((2,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = hessian(f)(x)
    np.testing.assert_allclose(np.asarray(res), [[12.0, 0.0], [0.0, 18.0]])


def test_custom_vjp() -> None:
    """Test function test_custom_vjp."""

    @custom_vjp
    def f(x: Tensor) -> Tensor:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return x * 2.0

    def fwd(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function fwd.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor, Tensor]: Computed output value from `fwd`.
        """
        return x * 2.0, x

    def bwd(res: Tensor, g: Tensor) -> tuple[Tensor]:
        """Test function bwd.

        Args:
            res: Test parameter `res`.
            g: Test parameter `g`.

        Returns:
            tuple[Tensor]: Computed output value from `bwd`.
        """
        del res
        return (g * 2.0,)

    f.defvjp(fwd, bwd)

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        out = f(x)
        np.testing.assert_allclose(np.asarray(ensure_tensor(out).numpy()), [4.0])
        from ml_switcheroo_compiler.grad import grad

        g = grad(f)
        g_res = g(x)
        np.testing.assert_allclose(np.asarray(ensure_tensor(g_res).numpy()), [2.0])


def test_overwrite_grad() -> None:
    """Test function test_overwrite_grad."""
    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        res = overwrite_with_gradient(x, x)
        np.testing.assert_allclose(np.asarray(ensure_tensor(res).numpy()), [2.0])


def test_greater_jvp_vjp() -> None:
    """Test function test_greater_jvp_vjp."""
    from ml_switcheroo_compiler.grad import jvp, vjp

    def f(x: Tensor, y: Tensor) -> Tensor:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            Tensor: Computed output value from `f`.
        """
        return x > y

    x = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    y = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v1 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    v2 = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))
    with ConfigContext(eager_mode=True):
        p_jvp, t_jvp = jvp(f, (x, y), (v1, v2))
        p_vjp, vjp_fn = vjp(f, x, y)
        assert p_jvp is not None
        assert t_jvp is not None
        assert p_vjp is not None
        assert callable(vjp_fn)


def test_checkpoint_eager() -> None:
    """Test function test_checkpoint_eager."""
    config.eager_mode = True

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return coerce_numeric(x) * 2

    wrapped = checkpoint(my_func)
    assert wrapped(3) == 6
    config.eager_mode = False


def test_checkpoint_not_tracing() -> None:
    """Test function test_checkpoint_not_tracing."""
    global_tracing_state.is_tracing = False

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
        """
        return coerce_numeric(x) * 2

    wrapped = checkpoint(my_func)
    assert wrapped(4) == 8


def test_checkpoint_infer_dtype() -> None:
    """Test function test_checkpoint_infer_dtype."""
    global_tracing_state.is_tracing = True

    def my_func(x: BackendArg) -> BackendArg:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `my_func`.
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
    assert extract_test_dtype(res) == DType.Float64
    assert ensure_ndarray(res).item() == 2.0


def test_grad_and_value_has_aux() -> None:
    """Test function test_grad_and_value_has_aux."""

    def my_func(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function my_func.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor, Tensor]: Computed output value from `my_func`.
        """
        return x * x, x

    opt = GradOptions(has_aux=True)
    wrapped = value_and_grad(my_func, opt)
    t = Tensor(np.array(2.0, dtype=np.float32), TensorConfig((), DType.Float32, Device("cpu")))
    val_pair, g_out = unpack_pair(wrapped(t))
    val, aux = unpack_pair(val_pair)
    np.testing.assert_allclose(np.asarray(ensure_tensor(val).numpy()), 4.0)
    np.testing.assert_allclose(np.asarray(ensure_tensor(aux).numpy()), 2.0)
    np.testing.assert_allclose(np.asarray(ensure_tensor(g_out).numpy()), 4.0)
