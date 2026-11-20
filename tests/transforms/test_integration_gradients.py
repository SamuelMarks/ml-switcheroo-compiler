"""Tests for test_integration_gradients."""

from __future__ import annotations

import typing

import numpy as np
import pytest

from tests.testing_utils import coerce_numeric, ensure_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_gradient_integration_across_backends() -> None:
    """Test function test_gradient_integration_across_backends."""
    from ml_switcheroo_compiler.core.config import config
    from ml_switcheroo_compiler.core.tensor import Tensor
    from ml_switcheroo_compiler.grad.api import grad

    def f(x: BackendArg) -> BackendArg:
        """Test function f.

        Args:
            x: Test parameter `x`.

        Returns:
            BackendArg: Computed output value from `f`.
        """
        return coerce_numeric(x) * coerce_numeric(x) + coerce_numeric(x) + coerce_numeric(x)

    grad_f = grad(f)
    x_val = np.array([3.0], dtype=np.float32)
    expected_grad = np.array([8.0], dtype=np.float32)
    backends_to_test = ["numpy", "pytorch", "jax"]
    from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
    from ml_switcheroo_compiler.core.tensor import TensorConfig
    from tests.testing_utils import adapt_backend_module

    results: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float64]]] = {}
    for be in backends_to_test:
        with pytest.raises(AttributeError):
            adapt_backend_module(config).require_symbol("set_backend")(be)
            x_t = Tensor(x_val, TensorConfig((3,), "float32", "cpu"))
            res = grad_f(x_t)
            results[be] = ensure_ndarray(ensure_tensor(res).numpy())
    for _be, res in results.items():
        np.testing.assert_allclose(res, expected_grad, rtol=1e-05)
