"""Tests for expanded sparse matrix formats (CSR, CSC), SpMM, SpGEMM, and gradient rules."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.sparse.kernels import (
    _apply_sparse_mask,
    _prepare_conv2d_tensors,
    csc_add,
    csr_add,
    dense_spmm,
    sparse_conv2d_mask,
    sparse_mask,
    spgemm,
    spgemm_grad,
    spmm,
    spmm_grad,
)
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


def test_dense_spmm_and_fallbacks() -> None:
    """Verify dense_spmm with sparse and dense operands."""
    dense_lhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dense_rhs = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # 1. Sparse operand
    csr_rhs = CSRTensor.from_dense(dense_rhs)
    res_csr = dense_spmm(dense_lhs, csr_rhs)
    np.testing.assert_allclose(res_csr, np.matmul(dense_lhs, dense_rhs))

    csc_rhs = CSCTensor.from_dense(dense_rhs)
    res_csc = dense_spmm(dense_lhs, csc_rhs)
    np.testing.assert_allclose(res_csc, np.matmul(dense_lhs, dense_rhs))

    coo_rhs = COOTensor.from_dense(dense_rhs)
    res_coo = dense_spmm(dense_lhs, coo_rhs)
    np.testing.assert_allclose(res_coo, np.matmul(dense_lhs, dense_rhs))

    # 2. Dense fallback operand (branch 746->752)
    res_dense = dense_spmm(dense_lhs, dense_rhs)
    np.testing.assert_allclose(res_dense, np.matmul(dense_lhs, dense_rhs))


def test_csr_add_and_csc_add() -> None:
    """Verify csr_add and csc_add with valid shapes, empty tensors, and fallbacks."""
    dense_a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    dense_b = np.array([[0.0, 3.0], [4.0, 0.0]], dtype=np.float32)

    csr_a = CSRTensor.from_dense(dense_a)
    csr_b = CSRTensor.from_dense(dense_b)
    empty_csr = CSRTensor.from_dense(np.zeros((2, 2), dtype=np.float32))

    # Empty operand branches
    assert csr_add(csr_a, empty_csr) is csr_a
    assert csr_add(empty_csr, csr_b) is csr_b

    # Normal addition
    res_csr = csr_add(csr_a, csr_b)
    np.testing.assert_allclose(res_csr.to_dense(), dense_a + dense_b)

    # Shape mismatch
    with pytest.raises(ValueError, match="Shape mismatch for CSR addition"):
        csr_add(csr_a, CSRTensor.from_dense(np.ones((3, 3), dtype=np.float32)))

    # Fallback branch when coo_add returns non-COOTensor
    with patch("ml_switcheroo_compiler.backends.sparse.kernels.coo_add", return_value=dense_a + dense_b):
        res_fb = csr_add(csr_a, csr_b)
        assert isinstance(res_fb, CSRTensor)
        np.testing.assert_allclose(res_fb.to_dense(), dense_a + dense_b)

    # CSC additions
    csc_a = CSCTensor.from_dense(dense_a)
    csc_b = CSCTensor.from_dense(dense_b)
    empty_csc = CSCTensor.from_dense(np.zeros((2, 2), dtype=np.float32))

    assert csc_add(csc_a, empty_csc) is csc_a
    assert csc_add(empty_csc, csc_b) is csc_b

    res_csc = csc_add(csc_a, csc_b)
    np.testing.assert_allclose(res_csc.to_dense(), dense_a + dense_b)

    with pytest.raises(ValueError, match="Shape mismatch for CSC addition"):
        csc_add(csc_a, CSCTensor.from_dense(np.ones((3, 3), dtype=np.float32)))

    # Fallback branch when coo_add returns non-COOTensor for CSC
    with patch("ml_switcheroo_compiler.backends.sparse.kernels.coo_add", return_value=dense_a + dense_b):
        res_csc_fb = csc_add(csc_a, csc_b)
        assert isinstance(res_csc_fb, CSCTensor)
        np.testing.assert_allclose(res_csc_fb.to_dense(), dense_a + dense_b)


def test_spgemm_dense_result_fallback() -> None:
    """Verify spgemm fallback branch when coo_matmat returns a dense array."""
    dense_a = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    dense_b = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    csr_a = CSRTensor.from_dense(dense_a)
    csr_b = CSRTensor.from_dense(dense_b)

    with patch("ml_switcheroo_compiler.backends.sparse.kernels.coo_matmat", return_value=np.matmul(dense_a, dense_b)):
        res = spgemm(csr_a, csr_b)
        assert isinstance(res, CSRTensor)
        np.testing.assert_allclose(res.to_dense(), np.matmul(dense_a, dense_b))


def test_sparse_mask_and_sparse_conv2d() -> None:
    """Verify sparse_mask and sparse_conv2d_mask operations with various shapes and masks."""
    data_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask_2d = np.array([[1, 0], [0, 1]], dtype=np.int32)

    # sparse_mask with COO and dense
    coo_data = COOTensor.from_dense(data_2d)
    res_m1 = sparse_mask(coo_data, mask_2d)
    assert isinstance(res_m1, COOTensor)
    np.testing.assert_allclose(res_m1.to_dense(), [[1.0, 0.0], [0.0, 4.0]])

    csr_mask = CSRTensor.from_dense(mask_2d)
    res_m2 = sparse_mask(data_2d, csr_mask)
    assert isinstance(res_m2, COOTensor)
    np.testing.assert_allclose(res_m2.to_dense(), [[1.0, 0.0], [0.0, 4.0]])

    # sparse_conv2d_mask with padding=(0, 0)
    x = np.ones((1, 1, 3, 3), dtype=np.float32)
    w = np.ones((1, 1, 2, 2), dtype=np.float32)
    mask_out = np.ones((2, 2), dtype=np.int32)

    conv_no_pad = sparse_conv2d_mask(x, w, mask_out, stride=(1, 1), padding=(0, 0))
    assert isinstance(conv_no_pad, COOTensor)
    assert conv_no_pad.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(conv_no_pad.to_dense(), np.full((1, 1, 2, 2), 4.0))

    # sparse_conv2d_mask with padding=(1, 1) and 2D inputs
    x_2d = np.ones((3, 3), dtype=np.float32)
    w_2d = np.ones((2, 2), dtype=np.float32)
    mask_out_pad = np.ones((4, 4), dtype=np.int32)
    conv_pad = sparse_conv2d_mask(x_2d, w_2d, mask_out_pad, stride=(1, 1), padding=(1, 1))
    assert isinstance(conv_pad, COOTensor)
    assert conv_pad.shape == (1, 1, 4, 4)

    # _prepare_conv2d_tensors with 3D weight
    w_3d = np.ones((1, 2, 2), dtype=np.float32)
    prep_x, prep_w = _prepare_conv2d_tensors(coo_data, w_3d)
    assert prep_x.ndim == 4
    assert prep_w.ndim == 4

    # _apply_sparse_mask with CSRTensor mask
    applied = _apply_sparse_mask(np.ones((1, 1, 2, 2), dtype=np.float32), csr_mask)
    assert isinstance(applied, COOTensor)
    assert applied.shape == (1, 1, 2, 2)
