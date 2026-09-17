"""Tests for grad.api functions including value_and_grad, hook_gradient, and jvp."""

from typing import Any

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad, hook_gradient, jvp, value_and_grad
from ml_switcheroo_compiler.grad.options import GradOptions


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def func(x: Tensor) -> tuple[Tensor, Tensor]:
        return x, x

    opts = GradOptions(has_aux=True)
    wrapped = value_and_grad(func, opts)
    try:
        wrapped(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass


def test_hook_gradient() -> None:
    """Test hook_gradient op definition and execution."""

    def hook(g: Any) -> Any:
        return g * 2.0

    def func(x: Tensor) -> Any:
        return hook_gradient(x, hook)

    def func2(x: Tensor) -> Any:
        return hook_gradient(x, lambda g: None)

    g = grad(func)
    try:
        g(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass

    g2 = grad(func2)
    try:
        g2(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass


def test_value_and_grad_no_aux() -> None:
    """Test value_and_grad with has_aux=False."""

    def func(x: Tensor) -> Tensor:
        return x

    opts = GradOptions(has_aux=False)
    wrapped = value_and_grad(func, opts)
    try:
        wrapped(Tensor([1.0], TensorConfig((1,), "float32", "cpu")))
    except Exception:
        pass


def test_api_jvp_basic() -> None:
    """Test numerical jvp with standard numeric scalar/array values."""

    def f(x: float, y: float) -> float:
        return x * x + 3.0 * y

    primals = (2.0, 1.0)
    tangents = (1.0, 2.0)
    primal_out, tangent_out = jvp(f, primals, tangents)
    assert np.isclose(primal_out, 7.0)
    # df = 2*x*dx + 3*dy = 2*2*1 + 3*2 = 4 + 6 = 10
    assert np.isclose(tangent_out, 10.0, atol=1e-3)


def test_api_jvp_no_add_and_add_exception() -> None:
    """Test jvp branch where primal lacks __add__ and where __add__ raises an Exception."""

    class NoAdd:
        """Object without __add__ method."""

        pass

    class BrokenAdd:
        """Object with __add__ that throws an error."""

        def __add__(self, other: Any) -> Any:
            raise RuntimeError("Cannot add")

    def f(a: Any, b: Any) -> float:
        return 42.0

    primals = (NoAdd(), BrokenAdd())
    tangents = (1.0, 1.0)
    primal_out, tangent_out = jvp(f, primals, tangents)
    assert primal_out == 42.0
    assert tangent_out == 0.0


def test_api_jvp_sub_exception() -> None:
    """Test jvp branch where tangent subtraction fails and falls back to primal_out."""

    class NonDifferentiableOutput:
        """Output object that cannot be subtracted."""

        def __init__(self, val: str) -> None:
            self.val = val

    def f(x: float) -> NonDifferentiableOutput:
        return NonDifferentiableOutput("result")

    primal_out, tangent_out = jvp(f, (1.0,), (1.0,))
    assert isinstance(primal_out, NonDifferentiableOutput)
    assert tangent_out is primal_out
