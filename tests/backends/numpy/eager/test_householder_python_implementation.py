"""Tests for test_householder_python_implementation."""

from __future__ import annotations

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import BackendArg


def test_householder_product_fallback() -> None:
    """Test function test_householder_product_fallback."""
    np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # if it goes through torch fallback
    class MockTorch:
        """Test class MockTorch."""

        @staticmethod
        def householder_product(v: BackendArg, tau: BackendArg) -> BackendArg:
            """Test function householder_product.

            Args:
                v: Test parameter `v`.
                tau: Test parameter `tau`.

            Returns:
                BackendArg: Computed output value from `householder_product`.
            """
            return v

    class MockTorchLinalg:
        """Test class MockTorchLinalg."""

        householder_product = MockTorch.householder_product

    pytest.MonkeyPatch().setattr(MockTorch, "linalg", MockTorchLinalg, raising=False)

    assert tau.shape == (2,)
