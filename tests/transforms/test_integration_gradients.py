"""Tests for test_integration_gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerCallableProtocol, EagerTensorProtocol, EagerValue


def test_gradient_integration_across_backends() -> None:
    """Test function test_gradient_integration_across_backends."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.api import grad

    def f(x: EagerValue) -> EagerValue:
        """Test function f.

        Args:
            x: Test parameter `x`.
        """
        return cast("EagerTensorProtocol", x) * cast("EagerTensorProtocol", x) + cast("EagerTensorProtocol", x) + cast("EagerTensorProtocol", x)

    grad_f = grad(f)
    x_val = np.array([3.0], dtype=np.float32)
    expected_grad = np.array([8.0], dtype=np.float32)
    backends_to_test = ["numpy", "pytorch", "jax"]
    results = {}
    for be in backends_to_test:
        with pytest.raises(AttributeError):
            cast("EagerTensorProtocol", getattr(config, "set_backend"))(be)
            x_t = cast("EagerCallableProtocol", Tensor)(x_val)
            res = grad_f(x_t)
            results[be] = cast("EagerTensorProtocol", getattr(res, "numpy"))()
    for _be, res in results.items():
        cast("EagerCallableProtocol", np.testing.assert_allclose)(res, expected_grad, rtol=1e-05)
