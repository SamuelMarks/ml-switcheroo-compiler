"""Tests for test_householder_product."""

from __future__ import annotations

import typing

import numpy as np

from tests.testing_utils import as_callable

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerValue


def test_householder_product_coverage() -> None:
    """Test function test_householder_product_coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _householder_product

    v = np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # Let's mock a backend_module with householder_product
    class MockBackend:
        """Test class MockBackend."""

        @staticmethod
        def householder_product(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
            """Test function householder_product.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.
            """
            return np.array([[1, 0], [0, 1]])

    assert as_callable(np.array_equal)(as_callable(_householder_product)(MockBackend, v, tau), v)

    class MockBackendLinalg:
        """Test class MockBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def householder_product(*args: EagerValue, **kwargs: EagerValue) -> EagerValue:
                """Test function householder_product.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.
                """
                return np.array([[1, 0], [0, 1]])

    assert as_callable(np.array_equal)(as_callable(_householder_product)(MockBackendLinalg, v, tau), v)
