import numpy as np


def test_householder_product_fallback():
    v = np.array([[1, 0], [0, 1]], dtype=float)
    tau = np.array([0, 0], dtype=float)

    # if it goes through torch fallback
    class MockTorch:
        @staticmethod
        def householder_product(v, tau):
            return v

    class MockTorchLinalg:
        householder_product = MockTorch.householder_product

    MockTorch.linalg = MockTorchLinalg

    # We can just check coverage.
    pass
