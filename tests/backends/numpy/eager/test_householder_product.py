import numpy as np


def test_householder_product_coverage():
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _householder_product

    v = np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # Let's mock a backend_module with householder_product
    class MockBackend:
        @staticmethod
        def householder_product(*args, **kwargs):
            return np.array([[1, 0], [0, 1]])

    assert np.array_equal(_householder_product(MockBackend, v, tau), v)

    class MockBackendLinalg:
        class linalg:
            @staticmethod
            def householder_product(*args, **kwargs):
                return np.array([[1, 0], [0, 1]])

    assert np.array_equal(_householder_product(MockBackendLinalg, v, tau), v)
