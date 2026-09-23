"""Tests for expanded sparse matrix formats (CSR, CSC), SpMM, SpGEMM, and gradient rules."""

from __future__ import annotations

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.sparse.kernels import spgemm, spgemm_grad, spmm, spmm_grad
from ml_switcheroo_compiler.backends.sparse.types import COOTensor, CSCTensor, CSRTensor


def test_csr_and_csc_tensor_conversions() -> None:
    """Verify CSR and CSC tensor representation, conversions, and roundtrips."""
    dense_2d = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [4.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )

    # CSR Tensor
    csr = CSRTensor.from_dense(dense_2d)
    assert csr.shape == (3, 3)
    assert csr.nnz == 5
    assert "<CSRTensor" in repr(csr)
    np.testing.assert_allclose(csr.to_dense(), dense_2d)

    # CSC Tensor
    csc = CSCTensor.from_dense(dense_2d)
    assert csc.shape == (3, 3)
    assert csc.nnz == 5
    assert "<CSCTensor" in repr(csc)
    np.testing.assert_allclose(csc.to_dense(), dense_2d)

    # Cross-conversions between COO, CSR, and CSC
    coo = COOTensor.from_dense(dense_2d)
    csr_from_coo = CSRTensor.from_coo(coo)
    np.testing.assert_allclose(csr_from_coo.to_dense(), dense_2d)

    csc_from_coo = CSCTensor.from_coo(coo)
    np.testing.assert_allclose(csc_from_coo.to_dense(), dense_2d)

    csc_from_coo_method = coo.to_csc()
    assert isinstance(csc_from_coo_method, CSCTensor)
    np.testing.assert_allclose(csc_from_coo_method.to_dense(), dense_2d)

    csr_from_csc = csc.to_csr()
    np.testing.assert_allclose(csr_from_csc.to_dense(), dense_2d)

    csc_from_csr = csr.to_csc()
    np.testing.assert_allclose(csc_from_csr.to_dense(), dense_2d)

    coo_from_csr = csr.to_coo()
    np.testing.assert_allclose(coo_from_csr.to_dense(), dense_2d)

    coo_from_csc = csc.to_coo()
    np.testing.assert_allclose(coo_from_csc.to_dense(), dense_2d)


def test_spmm_sparse_dense_matmul() -> None:
    """Verify sparse-dense matrix multiplication (SpMM) across CSR, CSC, and COO."""
    dense_a = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    dense_b = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float32,
    )
    expected = np.matmul(dense_a, dense_b)

    csr_a = CSRTensor.from_dense(dense_a)
    csc_a = CSCTensor.from_dense(dense_a)
    coo_a = COOTensor.from_dense(dense_a)

    # SpMM with 2D dense operand
    res_csr = spmm(csr_a, dense_b)
    np.testing.assert_allclose(res_csr, expected)

    res_csc = spmm(csc_a, dense_b)
    np.testing.assert_allclose(res_csc, expected)

    res_coo = spmm(coo_a, dense_b)
    np.testing.assert_allclose(res_coo, expected)

    # SpMM fallback with dense array operand
    res_dense = spmm(dense_a, dense_b)
    np.testing.assert_allclose(res_dense, expected)

    # SpMM with 1D dense vector operand
    vec_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected_vec = np.matmul(dense_a, vec_b)
    res_vec = spmm(csr_a, vec_b)
    np.testing.assert_allclose(res_vec, expected_vec)


def test_spgemm_sparse_sparse_matmul() -> None:
    """Verify sparse-sparse matrix multiplication (SpGEMM)."""
    dense_a = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    dense_b = np.array(
        [
            [0.0, 4.0],
            [5.0, 0.0],
            [0.0, 6.0],
        ],
        dtype=np.float32,
    )
    expected = np.matmul(dense_a, dense_b)

    csr_a = CSRTensor.from_dense(dense_a)
    csr_b = CSRTensor.from_dense(dense_b)

    prod_csr = spgemm(csr_a, csr_b)
    assert isinstance(prod_csr, CSRTensor)
    np.testing.assert_allclose(prod_csr.to_dense(), expected)

    # Shape mismatch error
    with pytest.raises(ValueError, match="Shape mismatch for SpGEMM"):
        spgemm(csr_a, CSRTensor.from_dense(np.ones((4, 4))))


def test_spmm_and_spgemm_gradients() -> None:
    """Verify gradient computation rules for SpMM and SpGEMM operations."""
    dense_a = np.array(
        [
            [2.0, 0.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    dense_b = np.array(
        [
            [1.0, 4.0],
            [2.0, 5.0],
        ],
        dtype=np.float32,
    )
    cotangent = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    # 1. SpMM gradients
    csr_a = CSRTensor.from_dense(dense_a)
    grad_a, grad_b = spmm_grad(csr_a, dense_b, cotangent)
    assert isinstance(grad_a, CSRTensor)
    assert isinstance(grad_b, np.ndarray)

    expected_grad_b = np.matmul(dense_a.T, cotangent)
    np.testing.assert_allclose(grad_b, expected_grad_b)

    expected_grad_a_dense = np.matmul(cotangent, dense_b.T) * (dense_a != 0)
    np.testing.assert_allclose(grad_a.to_dense(), expected_grad_a_dense)

    # 2. SpGEMM gradients
    csr_b = CSRTensor.from_dense(dense_b)
    grad_sp_a, grad_sp_b = spgemm_grad(csr_a, csr_b, cotangent)
    assert isinstance(grad_sp_a, CSRTensor)
    assert isinstance(grad_sp_b, CSRTensor)
    np.testing.assert_allclose(grad_sp_a.to_dense(), expected_grad_a_dense)
    np.testing.assert_allclose(grad_sp_b.to_dense(), expected_grad_b * (dense_b != 0))
