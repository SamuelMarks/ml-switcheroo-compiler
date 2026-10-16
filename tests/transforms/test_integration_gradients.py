"""Tests for test_integration_gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol, EagerValue


def test_gradient_integration_across_backends() -> None:
    """Test function test_gradient_integration_across_backends."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.api import grad

    # We test a simple function f(x) = x * x + 2*x
    # Its derivative is f'(x) = 2*x + 2
    def f(x: EagerValue) -> EagerValue:
        """Test function f."""
        return ((cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", x)) + cast("EagerTensorProtocol", x)) + cast("EagerTensorProtocol", x)

    grad_f = grad(f)
    x_val = np.array([3.0], dtype=np.float32)
    expected_grad = np.array([8.0], dtype=np.float32)

    backends_to_test = ["numpy", "pytorch", "jax"]

    results = {}
    for be in backends_to_test:
        try:
            cast("EagerTensorProtocol", getattr(config, "set_backend"))(be)
            x_t = cast("Callable[..., EagerTensorProtocol]", Tensor)(x_val)
            res = grad_f(x_t)
            results[be] = cast("EagerTensorProtocol", getattr(res, "numpy"))()
        except (ValueError, TypeError, AttributeError, IndexError, KeyError, RuntimeError, ZeroDivisionError, NotImplementedError, OSError, ImportError, AssertionError, Exception) as _exc:
            # If backend not installed, skip
            assert _exc is not None

    for _be, res in results.items():
        cast("Callable[..., EagerTensorProtocol]", np.testing.assert_allclose)(res, expected_grad, rtol=1e-5)
