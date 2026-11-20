"""Tests for test_householder_product."""

from __future__ import annotations

import numpy as np

from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from tests.testing_utils import BackendArg, MockTestStub


def test_householder_product_coverage() -> None:
    """Test function test_householder_product_coverage."""
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _householder_product

    v = np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # Let's mock a backend_module with householder_product
    class MockBackend(MockTestStub):
        """Test class MockBackend."""

        @staticmethod
        def householder_product(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
            """Test function householder_product.

            Args:
                args: Test parameter `args`.
                kwargs: Test parameter `kwargs`.

            Returns:
                BackendArg: Computed output value from `householder_product`.
            """
            return np.array([[1, 0], [0, 1]])

    assert np.array_equal(ensure_ndarray(_householder_product(MockBackend(), v, tau)), v)

    class MockBackendLinalg(MockTestStub):
        """Test class MockBackendLinalg."""

        class linalg:
            """Test class linalg."""

            @staticmethod
            def householder_product(*args: BackendArg, **kwargs: BackendArg) -> BackendArg:
                """Test function householder_product.

                Args:
                    args: Test parameter `args`.
                    kwargs: Test parameter `kwargs`.

                Returns:
                    BackendArg: Computed output value from `householder_product`.
                """
                return np.array([[1, 0], [0, 1]])

    assert np.array_equal(ensure_ndarray(_householder_product(MockBackendLinalg(), v, tau)), v)
