"""Fuzzing and equivalence tests for higher-order derivatives and Hessian-vector products."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

import ml_switcheroo_compiler.ops as ops
from ml_switcheroo_compiler.backends.registry import get_active_backend
from ml_switcheroo_compiler.core.device import Device
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.grad import grad, hessian, hvp

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue

pytest.importorskip("hypothesis")


def make_tensor(val: float) -> Tensor:
    """Create a 1D float32 Tensor.

    Args:
        val (float): Scalar float value.

    Returns:
        Tensor: Constructed tensor.
    """
    return Tensor(np.array([val], dtype=np.float32), TensorConfig((1,), DType.Float32, Device("cpu")))


@settings(max_examples=10, deadline=None)
@given(val=st.floats(min_value=-1.8, max_value=1.8, allow_nan=False, allow_infinity=False))
def test_hvp_polynomial_vs_finite_difference(val: float) -> None:
    """Test HVP for polynomial f(x) = x^3 against central finite difference of gradients."""
    eps = 1e-3

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""
        return ops.multiply(x, ops.multiply(x, x))

    x = make_tensor(val)
    v = make_tensor(1.0)

    # Compute HVP using compiler engine
    _, tan = cast("Callable[..., EagerTensorProtocol]", hvp)(f, (x,), (v,))
    computed_hvp = cast("Callable[..., EagerTensorProtocol]", float)(cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(tan), "__getitem__")(0)))

    # Central finite difference of gradients: (g(x + eps) - g(x - eps)) / (2 * eps)
    g_fn = grad(f)
    g_plus = cast("Callable[..., EagerTensorProtocol]", float)(
        cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(cast("Callable[..., EagerTensorProtocol]", g_fn)(cast("Callable[..., EagerTensorProtocol]", make_tensor)(val + eps))), "__getitem__")(0))
    )
    g_minus = cast("Callable[..., EagerTensorProtocol]", float)(
        cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(cast("Callable[..., EagerTensorProtocol]", g_fn)(cast("Callable[..., EagerTensorProtocol]", make_tensor)(val - eps))), "__getitem__")(0))
    )
    fd_hvp = (g_plus - g_minus) / (2.0 * eps)

    # Analytical expected: d^2/dx^2 (x^3) = 6 * x
    analytical_hvp = 6.0 * val

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(computed_hvp, fd_hvp, rtol=1e-2, atol=1e-2)


@settings(max_examples=10, deadline=None)
@given(val=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False))
def test_hvp_trigonometric_vs_finite_difference(val: float) -> None:
    """Test HVP for f(x) = sin(x) against analytical and finite difference."""
    eps = 1e-3

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""
        return ops.sin(x)

    x = make_tensor(val)
    v = make_tensor(1.0)

    _, tan = cast("Callable[..., EagerTensorProtocol]", hvp)(f, (x,), (v,))
    computed_hvp = cast("Callable[..., EagerTensorProtocol]", float)(cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(tan), "__getitem__")(0)))

    g_fn = grad(f)
    g_plus = cast("Callable[..., EagerTensorProtocol]", float)(
        cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(cast("Callable[..., EagerTensorProtocol]", g_fn)(cast("Callable[..., EagerTensorProtocol]", make_tensor)(val + eps))), "__getitem__")(0))
    )
    g_minus = cast("Callable[..., EagerTensorProtocol]", float)(
        cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(cast("Callable[..., EagerTensorProtocol]", g_fn)(cast("Callable[..., EagerTensorProtocol]", make_tensor)(val - eps))), "__getitem__")(0))
    )
    fd_hvp = (g_plus - g_minus) / (2.0 * eps)

    # Analytical: d^2/dx^2 sin(x) = -sin(x)
    analytical_hvp = -float(np.sin(val))

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(computed_hvp, fd_hvp, rtol=1e-2, atol=1e-2)


@settings(max_examples=10, deadline=None)
@given(val=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_hvp_exponential_vs_analytical(val: float) -> None:
    """Test HVP for f(x) = exp(x) against analytical d^2/dx^2 exp(x) = exp(x)."""

    def f(x: EagerValue) -> EagerValue:
        """Test function f."""
        return ops.exp(x)

    x = make_tensor(val)
    v = make_tensor(1.0)

    _, tan = cast("Callable[..., EagerTensorProtocol]", hvp)(f, (x,), (v,))
    computed_hvp = cast("Callable[..., EagerTensorProtocol]", float)(cast("EagerTensorProtocol", getattr(cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(tan), "__getitem__")(0)))
    analytical_hvp = float(np.exp(val))

    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)


def test_hessian_matrix_analytical() -> None:
    """Verify full Hessian computation for multi-step scalar function."""

    def f(x: EagerValue) -> EagerValue:
        # f(x) = x^4, H(x) = 12 * x^2
        """Test function f."""
        x2 = ops.multiply(x, x)
        return ops.multiply(x2, x2)

    val = 2.0
    x = make_tensor(val)
    h_fn = hessian(f)
    h_mat = cast("Callable[..., EagerTensorProtocol]", get_active_backend)().asarray(cast("Callable[..., EagerTensorProtocol]", h_fn)(x))

    expected = np.array([[12.0 * (val**2)]], dtype=np.float32)
    cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(h_mat, expected, rtol=1e-3, atol=1e-3)
