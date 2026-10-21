"""Tests for test_integration_gradients."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


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
        return as_tensor(x) * as_tensor(x) + as_tensor(x) + as_tensor(x)

    grad_f = grad(f)
    x_val = np.array([3.0], dtype=np.float32)
    expected_grad = np.array([8.0], dtype=np.float32)
    backends_to_test = ["numpy", "pytorch", "jax"]
    results = {}
    for be in backends_to_test:
        with pytest.raises(AttributeError):
            as_tensor(getattr(config, "set_backend"))(be)
            x_t = as_callable(Tensor)(x_val)
            res = grad_f(x_t)
            results[be] = as_tensor(getattr(res, "numpy"))()
    for _be, res in results.items():
        as_callable(np.testing.assert_allclose)(res, expected_grad, rtol=1e-05)
