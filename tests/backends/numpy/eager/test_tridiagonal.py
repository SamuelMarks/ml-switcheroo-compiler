import numpy as np


def test_tridiagonal_coverage():
    from ml_switcheroo_compiler.backends.eager.core_math_ops import _mock_tridiagonal

    alpha = np.array([2, 2, 2], dtype=float)

    class MockBackend:
        @staticmethod
        def tridiagonal(*args, **kwargs):
            return np.array([1, 2, 3])

    assert np.array_equal(_mock_tridiagonal(MockBackend, alpha), np.array([1, 2, 3]))

    class MockBackendRandom:
        class random:
            @staticmethod
            def tridiagonal(*args, **kwargs):
                return np.array([4, 5, 6])

    assert np.array_equal(_mock_tridiagonal(MockBackendRandom, alpha), np.array([4, 5, 6]))
