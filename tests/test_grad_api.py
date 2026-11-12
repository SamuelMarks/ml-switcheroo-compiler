"""Tests for grad.api functions including value_and_grad, hook_gradient, and jvp."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad, hook_gradient, jvp, value_and_grad
from ml_switcheroo_compiler.grad.options import GradOptions
from tests.testing_utils import ensure_callable, ensure_tensor_obj

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def func(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function func.

        Args:
            x: Test parameter `x`.

        Returns:
            tuple[Tensor, Tensor]: Computed output value from `func`.
        """
        return x, x

    opts = GradOptions(has_aux=True)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    (val, aux), g_out = ensure_callable(wrapped)(t_in)
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(val).numpy()), [1.0])
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(aux).numpy()), [1.0])
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(g_out).numpy()), [1.0])


def test_hook_gradient() -> None:
    """Test hook_gradient op definition and execution."""

    def hook(g: Tensor) -> Tensor:
        """Test function hook.

        Args:
            g: Test parameter `g`.

        Returns:
            Tensor: Computed output value from `hook`.
        """
        return g * 2.0

    def func(x: Tensor) -> Tensor:
        """Test function func.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `func`.
        """
        return ensure_tensor_obj(hook_gradient(x, hook))

    def func2(x: Tensor) -> Tensor:
        """Test function func2.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `func2`.
        """
        return ensure_tensor_obj(hook_gradient(x, lambda g: None))

    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    g = grad(func)
    res1 = g(t_in)
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(res1).numpy()), [2.0])

    g2 = grad(func2)
    res2 = g2(t_in)
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(res2).numpy()), [1.0])


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def func(x: Tensor) -> Tensor:
        """Test function func.

        Args:
            x: Test parameter `x`.

        Returns:
            Tensor: Computed output value from `func`.
        """
        return x

    opts = GradOptions(has_aux=False)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    val, g_out = ensure_callable(wrapped)(t_in)
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(val).numpy()), [1.0])
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(g_out).numpy()), [1.0])


def test_api_jvp_basic() -> None:
    """Test symbolic and fallback jvp with standard numeric scalar/array values."""
    from unittest.mock import patch

    def f(x: float, y: float) -> float:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.

        Returns:
            float: Computed output value from `f`.
        """
        return x * x + 3.0 * y

    primals = (2.0, 1.0)
    tangents = (1.0, 2.0)
    primal_out, tangent_out = jvp(f, primals, tangents)
    assert ensure_callable(np.isclose)(primal_out, 7.0)
    assert ensure_callable(np.isclose)(tangent_out, 10.0, atol=1e-5)

    t_p = Tensor(np.array([2.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    t_t = Tensor(np.array([3.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    p_ten, t_ten = jvp(lambda x: x * x, (t_p,), (t_t,))
    np.testing.assert_allclose(np.asarray(ensure_tensor_obj(p_ten).numpy()), [4.0])
    np.testing.assert_allclose(np.asarray(t_ten), [12.0])

    # Also test fallback when _symbolic_jvp returns non-pair or raises RuntimeError, while propagating AutodiffError
    with patch("ml_switcheroo_compiler.grad.jvp_vjp.jvp", return_value="not_a_tuple"):
        p_out2, t_out2 = jvp(f, primals, tangents)
        assert ensure_callable(np.isclose)(p_out2, 7.0)
        assert ensure_callable(np.isclose)(t_out2, 10.0, atol=1e-3)

    from ml_switcheroo_compiler.core.errors import AutodiffError

    with patch("ml_switcheroo_compiler.grad.jvp_vjp.jvp", side_effect=AutodiffError("missing rule")):
        with pytest.raises(AutodiffError, match="missing rule"):
            jvp(f, primals, tangents)


def test_api_jvp_no_add_and_add_exception() -> None:
    """Test jvp leaves non-addable auxiliary primals unchanged and raises AutodiffError on broken __add__."""
    from ml_switcheroo_compiler.core.errors import AutodiffError

    class NoAdd:
        """Object without __add__ method."""

    class BrokenAdd:
        """Object with __add__ that throws an error."""

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.

            Returns:
                EagerValue: Computed output value from `__add__`.
            """
            raise RuntimeError("Cannot add")

    def f(a: EagerValue, b: EagerValue) -> float:
        """Test function f.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.

        Returns:
            float: Computed output value from `f`.
        """
        _ = (a, b)
        return 42.0

    primal_out, tangent_out = ensure_callable(jvp)(f, (NoAdd(), 1.0), (1.0, 0.0))
    assert primal_out == 42.0
    assert tangent_out == 0.0

    with pytest.raises(AutodiffError, match="Failed to perturb primal operand"):
        ensure_callable(jvp)(f, (NoAdd(), BrokenAdd()), (1.0, 1.0))


def test_api_jvp_sub_exception() -> None:
    """Test jvp raises AutodiffError when function output does not support tangent subtraction."""
    from ml_switcheroo_compiler.core.errors import AutodiffError

    class NonDifferentiableOutput:
        """Output object that cannot be subtracted."""

        def __init__(self, val: str) -> None:
            """Test function __init__.

            Args:
                val: Test parameter `val`.
            """
            self.val = val

    def f(x: float) -> NonDifferentiableOutput:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            NonDifferentiableOutput: Computed output value from `f`.
        """
        _ = x
        return NonDifferentiableOutput("result")

    with pytest.raises(AutodiffError, match="Failed to compute directional derivative tangent"):
        ensure_callable(jvp)(f, (1.0,), (1.0,))
