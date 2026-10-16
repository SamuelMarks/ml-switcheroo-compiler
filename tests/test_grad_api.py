"""Tests for grad.api functions including value_and_grad, hook_gradient, and jvp."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad.api import grad, hook_gradient, jvp, value_and_grad
from ml_switcheroo_compiler.grad.options import GradOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_value_and_grad_has_aux() -> None:
    """Test value_and_grad with has_aux=True."""

    def func(x: Tensor) -> tuple[Tensor, Tensor]:
        """Test function func."""
        return x, x

    opts = GradOptions(has_aux=True)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    (val, aux), g_out = cast("Callable[..., EagerTensorProtocol]", wrapped)(t_in)
    assert val is not None
    assert aux is not None
    assert g_out is not None


def test_hook_gradient() -> None:
    """Test hook_gradient op definition and execution."""

    def hook(g: Tensor) -> Tensor:
        """Test function hook."""
        return g * 2.0

    def func(x: Tensor) -> Tensor:
        """Test function func."""
        return cast("Tensor", hook_gradient(x, hook))

    def func2(x: Tensor) -> Tensor:
        """Test function func2."""
        return cast("Tensor", hook_gradient(x, lambda g: None))

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
        """Test function func."""
        return x

    opts = GradOptions(has_aux=False)
    wrapped = value_and_grad(func, opts)
    t_in = Tensor(np.array([1.0], dtype=np.float32), TensorConfig((1,), "float32", "cpu"))
    val, g_out = cast("Callable[..., EagerTensorProtocol]", wrapped)(t_in)
    assert val is not None
    assert g_out is not None


def test_api_jvp_basic() -> None:
    """Test numerical jvp with standard numeric scalar/array values."""

    def f(x: float, y: float) -> float:
        """Test function f."""
        return x * x + 3.0 * y

    primals = (2.0, 1.0)
    tangents = (1.0, 2.0)
    primal_out, tangent_out = jvp(f, primals, tangents)
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(primal_out, 7.0)
    assert cast("Callable[..., EagerTensorProtocol]", np.isclose)(tangent_out, 10.0, atol=1e-3)


def test_api_jvp_no_add_and_add_exception() -> None:
    """Test jvp branch where primal lacks __add__ and where __add__ raises an Exception."""

    class NoAdd:
        """Object without __add__ method."""

    class BrokenAdd:
        """Object with __add__ that throws an error."""

        def __add__(self, other: EagerValue) -> EagerValue:
            """Test function __add__."""
            raise RuntimeError("Cannot add")

    def f(a: EagerValue, b: EagerValue) -> float:
        """Test function f."""
        _ = (a, b)
        return 42.0

    primals = (NoAdd(), BrokenAdd())
    tangents = (1.0, 1.0)
    primal_out, tangent_out = cast("Callable[..., EagerTensorProtocol]", jvp)(f, primals, tangents)
    assert primal_out == 42.0
    assert tangent_out == 0.0


def test_api_jvp_sub_exception() -> None:
    """Test jvp branch where tangent subtraction fails and falls back to primal_out."""

    class NonDifferentiableOutput:
        """Output object that cannot be subtracted."""

        def __init__(self, val: str) -> None:
            """Test function __init__."""
            self.val = val

    def f(x: float) -> NonDifferentiableOutput:
        """Test function f."""
        _ = x
        return NonDifferentiableOutput("result")

    primal_out, tangent_out = cast("Callable[..., EagerTensorProtocol]", jvp)(f, (1.0,), (1.0,))
    assert isinstance(primal_out, NonDifferentiableOutput)
    assert tangent_out is primal_out
