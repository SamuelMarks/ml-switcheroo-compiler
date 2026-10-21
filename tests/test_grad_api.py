"""Tests for grad.api functions including value_and_grad, hook_gradient, and jvp."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad, hook_gradient, jvp, value_and_grad
from ml_switcheroo_compiler.grad.options import GradOptions
from tests.testing_utils import as_callable, as_tensor_obj

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def func(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function func.

        Args:
            x: Test parameter `x`.
        """
        return x, x

    opts = GradOptions(has_aux=True)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    (val, aux), g_out = as_callable(wrapped)(t_in)
    assert val is not None
    assert aux is not None
    assert g_out is not None


def test_hook_gradient() -> None:
    """Test hook_gradient op definition and execution."""

    def hook(g: Tensor) -> Tensor:
        """Test function hook.

        Args:
            g: Test parameter `g`.
        """
        return g * 2.0

    def func(x: Tensor) -> Tensor:
        """Test function func.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor_obj(hook_gradient(x, hook))

    def func2(x: Tensor) -> Tensor:
        """Test function func2.

        Args:
            x: Test parameter `x`.
        """
        return as_tensor_obj(hook_gradient(x, lambda g: None))

    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    g = grad(func)
    res1 = g(t_in)
    assert res1 is not None

    g2 = grad(func2)
    res2 = g2(t_in)
    assert res2 is not None


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def func(x: Tensor) -> Tensor:
        """Test function func.

        Args:
            x: Test parameter `x`.
        """
        return x

    opts = GradOptions(has_aux=False)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    val, g_out = as_callable(wrapped)(t_in)
    assert val is not None
    assert g_out is not None


def test_api_jvp_basic() -> None:
    """Test numerical jvp with standard numeric scalar/array values."""

    def f(x: float, y: float) -> float:
        """Test function f.

        Args:
            x: Test parameter `x`.
            y: Test parameter `y`.
        """
        return x * x + 3.0 * y

    primals = (2.0, 1.0)
    tangents = (1.0, 2.0)
    primal_out, tangent_out = jvp(f, primals, tangents)
    assert as_callable(np.isclose)(primal_out, 7.0)
    assert as_callable(np.isclose)(tangent_out, 10.0, atol=1e-3)


def test_api_jvp_no_add_and_add_exception() -> None:
    """Test jvp branch where primal lacks __add__ and where __add__ raises an Exception."""

    class NoAdd:
        """Object without __add__ method."""

    class BrokenAdd:
        """Object with __add__ that throws an error."""

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__.

            Args:
                other: Test parameter `other`.
            """
            raise RuntimeError("Cannot add")

    def f(a: EagerValue, b: EagerValue) -> float:
        """Test function f.

        Args:
            a: Test parameter `a`.
            b: Test parameter `b`.
        """
        _ = (a, b)
        return 42.0

    primals = (NoAdd(), BrokenAdd())
    tangents = (1.0, 1.0)
    primal_out, tangent_out = as_callable(jvp)(f, primals, tangents)
    assert primal_out == 42.0
    assert tangent_out == 0.0


def test_api_jvp_sub_exception() -> None:
    """Test jvp branch where tangent subtraction fails and falls back to primal_out."""

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
        """
        _ = x
        return NonDifferentiableOutput("result")

    primal_out, tangent_out = as_callable(jvp)(f, (1.0,), (1.0,))
    assert isinstance(primal_out, NonDifferentiableOutput)
    assert tangent_out is primal_out
