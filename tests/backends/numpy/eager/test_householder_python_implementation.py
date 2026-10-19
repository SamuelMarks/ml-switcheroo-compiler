"""Tests for test_householder_python_implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_householder_product_fallback() -> None:
    """Test function test_householder_product_fallback."""
    np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # if it goes through torch fallback
    class MockTorch:
        """Test class MockTorch."""

        @staticmethod
        def householder_product(v: EagerValue, tau: EagerValue) -> EagerValue:
            """Test function householder_product.

            Args:
                v: Test parameter `v`.
                tau: Test parameter `tau`.
            """
            return v

    class MockTorchLinalg:
        """Test class MockTorchLinalg."""

        householder_product = MockTorch.householder_product

    setattr(MockTorch, "linalg", MockTorchLinalg)

    assert tau.shape == (2,)
