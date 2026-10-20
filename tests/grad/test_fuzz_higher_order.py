"""Fuzzing and equivalence tests for higher-order derivatives and Hessian-vector products."""

from __future__ import annotations

import typing

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

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue

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
    """Test HVP for polynomial f(x) = x^3 against central finite difference of gradients.

    Args:
        val: Test parameter `val`.
    """
    eps = 1e-3

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return ops.multiply(x, ops.multiply(x, x))

    x = make_tensor(val)
    v = make_tensor(1.0)

    # Compute HVP using compiler engine
    _, tan = typing.cast("EagerCallableProtocol", hvp)(f, (x,), (v,))
    computed_hvp = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(tan), "__getitem__")(0)))

    # Central finite difference of gradients: (g(x + eps) - g(x - eps)) / (2 * eps)
    g_fn = grad(f)
    g_plus = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(typing.cast("EagerCallableProtocol", g_fn)(typing.cast("EagerCallableProtocol", make_tensor)(val + eps))), "__getitem__")(0)))
    g_minus = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(typing.cast("EagerCallableProtocol", g_fn)(typing.cast("EagerCallableProtocol", make_tensor)(val - eps))), "__getitem__")(0)))
    fd_hvp = (g_plus - g_minus) / (2.0 * eps)

    # Analytical expected: d^2/dx^2 (x^3) = 6 * x
    analytical_hvp = 6.0 * val

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(computed_hvp, fd_hvp, rtol=1e-2, atol=1e-2)


@settings(max_examples=10, deadline=None)
@given(val=st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False))
def test_hvp_trigonometric_vs_finite_difference(val: float) -> None:
    """Test HVP for f(x) = sin(x) against analytical and finite difference.

    Args:
        val: Test parameter `val`.
    """
    eps = 1e-3

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return ops.sin(x)

    x = make_tensor(val)
    v = make_tensor(1.0)

    _, tan = typing.cast("EagerCallableProtocol", hvp)(f, (x,), (v,))
    computed_hvp = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(tan), "__getitem__")(0)))

    g_fn = grad(f)
    g_plus = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(typing.cast("EagerCallableProtocol", g_fn)(typing.cast("EagerCallableProtocol", make_tensor)(val + eps))), "__getitem__")(0)))
    g_minus = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(typing.cast("EagerCallableProtocol", g_fn)(typing.cast("EagerCallableProtocol", make_tensor)(val - eps))), "__getitem__")(0)))
    fd_hvp = (g_plus - g_minus) / (2.0 * eps)

    # Analytical: d^2/dx^2 sin(x) = -sin(x)
    analytical_hvp = -float(np.sin(val))

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(computed_hvp, fd_hvp, rtol=1e-2, atol=1e-2)


@settings(max_examples=10, deadline=None)
@given(val=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_hvp_exponential_vs_analytical(val: float) -> None:
    """Test HVP for f(x) = exp(x) against analytical d^2/dx^2 exp(x) = exp(x).

    Args:
        val: Test parameter `val`.
    """

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return ops.exp(x)

    x = make_tensor(val)
    v = make_tensor(1.0)

    _, tan = typing.cast("EagerCallableProtocol", hvp)(f, (x,), (v,))
    computed_hvp = typing.cast("EagerCallableProtocol", float)(typing.cast("EagerTensorProtocol", getattr(typing.cast("EagerCallableProtocol", get_active_backend)().asarray(tan), "__getitem__")(0)))
    analytical_hvp = float(np.exp(val))

    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(computed_hvp, analytical_hvp, rtol=1e-2, atol=1e-2)


def test_hessian_matrix_analytical() -> None:
    """Verify full Hessian computation for multi-step scalar function."""

    def f(x: EagerValue) -> EagerValue:
        # f(x) = x^4, H(x) = 12 * x^2
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        x2 = ops.multiply(x, x)
        return ops.multiply(x2, x2)

    val = 2.0
    x = make_tensor(val)
    h_fn = hessian(f)
    h_mat = typing.cast("EagerCallableProtocol", get_active_backend)().asarray(typing.cast("EagerCallableProtocol", h_fn)(x))

    expected = np.array([[12.0 * (val**2)]], dtype=np.float32)
    typing.cast("EagerCallableProtocol", np.testing.assert_allclose)(h_mat, expected, rtol=1e-3, atol=1e-3)
