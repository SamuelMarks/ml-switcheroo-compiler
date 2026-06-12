"""Unit tests for basic linear algebra operations including matrix multiplication, dot.

product, and Einstein summation.
"""

import numpy as np

from ml_switcheroo.ops.linalg.basic import (
    Dot,
    Einsum,
    Matmul,
)


def test_matmul_op() -> None:
    """Tests the matrix multiplication operator.

    This test verifies that the Matmul operator correctly infers the output shape
    of two matrices and evaluates the matrix multiplication using NumPy's matmul
    implementation

    Returns:
    None
    """
    op = Matmul()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)

    assert op.infer_shape(a.shape, b.shape) == (2, 4)
    assert op.infer_shape(None, None) is None

    res = op.numpy_eval(a, b)
    assert np.allclose(res, np.matmul(a, b))





def test_dot_op() -> None:
    """Tests the dot product operator.

    This test verifies that the Dot operator correctly handles shape inference
    and evaluates the dot product of two 1D arrays using NumPy's dot
    implementation

    Returns:
    None
    """
    op = Dot()
    a = np.random.randn(3)
    b = np.random.randn(3)

    assert op.infer_shape(a.shape, b.shape) is None

    res = op.numpy_eval(a, b)
    assert np.allclose(res, np.dot(a, b))





def test_einsum_op() -> None:
    """Tests the Einstein summation operator.

    This test verifies that the Einsum operator correctly handles shape inference
    and evaluates the Einstein summation of two matrices using NumPy's einsum
    implementation with a specified subscript string

    Returns:
    None
    """
    op = Einsum()
    a = np.random.randn(2, 3)
    b = np.random.randn(3, 4)
    subscripts = "ij,jk->ik"

    assert op.infer_shape(subscripts, a.shape, b.shape) is None

    res = op.numpy_eval(subscripts, a, b)
    assert np.allclose(res, np.einsum(subscripts, a, b))
