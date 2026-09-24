"""Unit tests for Sparse backend kernel operations and format conversions."""

from __future__ import annotations

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.sparse as sp_pkg
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.sparse.types import COOTensor, CSCTensor, CSRTensor


def test_sparse_conversions_and_kernels() -> None:
    """Verify Sparse format inter-conversions, SpMM, SpGEMM, addition, and convolution masks."""
    dense_mat = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]], dtype=np.float32)

    # Direct format inter-conversions
    coo = COOTensor.from_dense(dense_mat)
    csr = sp_pkg.coo_to_csr(coo)
    csc = sp_pkg.coo_to_csc(coo)

    np.testing.assert_allclose(csr.to_dense(), dense_mat)
    np.testing.assert_allclose(csc.to_dense(), dense_mat)

    # CSR <-> CSC and COO roundtrips
    coo_from_csr = sp_pkg.csr_to_coo(csr)
    np.testing.assert_allclose(coo_from_csr.to_dense(), dense_mat)

    csc_from_csr = sp_pkg.csr_to_csc(csr)
    np.testing.assert_allclose(csc_from_csr.to_dense(), dense_mat)

    coo_from_csc = sp_pkg.csc_to_coo(csc)
    np.testing.assert_allclose(coo_from_csc.to_dense(), dense_mat)

    csr_from_csc = sp_pkg.csc_to_csr(csc)
    np.testing.assert_allclose(csr_from_csc.to_dense(), dense_mat)

    # Non-2D error checking
    coo_1d = COOTensor.from_dense(np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="CSR conversion requires 2D tensor"):
        sp_pkg.coo_to_csr(coo_1d)
    with pytest.raises(ValueError, match="CSC conversion requires 2D tensor"):
        sp_pkg.coo_to_csc(coo_1d)

    # Empty tensor conversions
    empty_coo = COOTensor(indices=np.empty((2, 0), dtype=np.int64), values=np.empty((0,), dtype=np.float32), shape=(3, 3))
    empty_csr = sp_pkg.coo_to_csr(empty_coo)
    assert empty_csr.nnz == 0
    empty_csc = sp_pkg.coo_to_csc(empty_coo)
    assert empty_csc.nnz == 0

    # SpMM and dense_spmm
    dense_rhs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    expected_prod = dense_mat @ dense_rhs

    spmm_res = sp_pkg.spmm(csr, dense_rhs)
    np.testing.assert_allclose(spmm_res, expected_prod)

    spmm_csc_res = sp_pkg.spmm(csc, dense_rhs)
    np.testing.assert_allclose(spmm_csc_res, expected_prod)

    dense_lhs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    expected_dense_spmm = dense_lhs @ dense_mat
    dense_spmm_res = sp_pkg.dense_spmm(dense_lhs, csr)
    np.testing.assert_allclose(dense_spmm_res, expected_dense_spmm)

    # SpGEMM
    dense_b = np.array([[0.0, 2.0, 0.0], [1.0, 0.0, 3.0], [0.0, 4.0, 0.0]], dtype=np.float32)
    csr_b = CSRTensor.from_dense(dense_b)
    expected_spgemm = dense_mat @ dense_b
    spgemm_res = sp_pkg.spgemm(csr, csr_b)
    np.testing.assert_allclose(spgemm_res.to_dense(), expected_spgemm)

    bad_b = CSRTensor.from_dense(np.ones((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="Shape mismatch for SpGEMM"):
        sp_pkg.spgemm(csr, bad_b)

    # Elementwise addition (CSR and CSC)
    expected_add = dense_mat + dense_b
    csr_add_res = sp_pkg.csr_add(csr, csr_b)
    np.testing.assert_allclose(csr_add_res.to_dense(), expected_add)

    csc_b = CSCTensor.from_dense(dense_b)
    csc_add_res = sp_pkg.csc_add(csc, csc_b)
    np.testing.assert_allclose(csc_add_res.to_dense(), expected_add)

    # Empty operand branches for addition
    empty_csr_zero = CSRTensor.from_dense(np.zeros((3, 3), dtype=np.float32))
    assert sp_pkg.csr_add(csr, empty_csr_zero) is csr
    assert sp_pkg.csr_add(empty_csr_zero, csr) is csr

    empty_csc_zero = CSCTensor.from_dense(np.zeros((3, 3), dtype=np.float32))
    assert sp_pkg.csc_add(csc, empty_csc_zero) is csc
    assert sp_pkg.csc_add(empty_csc_zero, csc) is csc

    with pytest.raises(ValueError, match="Shape mismatch for CSR addition"):
        sp_pkg.csr_add(csr, bad_b)
    with pytest.raises(ValueError, match="Shape mismatch for CSC addition"):
        sp_pkg.csc_add(csc, CSCTensor.from_dense(np.ones((4, 4), dtype=np.float32)))

    # Sparse convolution masks with 2D, 3D and padding
    x_2d = np.ones((4, 4), dtype=np.float32)
    w_2d = np.ones((3, 3), dtype=np.float32)
    mask_pad = np.ones((4, 4), dtype=np.int64)
    conv_2d_res = sp_pkg.sparse_conv2d_mask(x_2d, w_2d, mask_pad, padding=(1, 1))
    assert conv_2d_res is not None

    x_3d = np.ones((1, 4, 4), dtype=np.float32)
    w_3d = np.ones((1, 3, 3), dtype=np.float32)
    mask_3d = np.ones((2, 2), dtype=np.int64)
    conv_3d_res = sp_pkg.sparse_conv2d_mask(x_3d, w_3d, mask_3d)
    assert conv_3d_res is not None

    # Sparse convolution masks and sparse_mask
    mask = np.array([[1, 0], [0, 1]], dtype=np.int64)
    tensor_to_mask = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    s_mask_res = sp_pkg.sparse_mask(tensor_to_mask, mask)
    assert s_mask_res.nnz == 2
    np.testing.assert_allclose(s_mask_res.to_dense(), [[10.0, 0.0], [0.0, 40.0]])

    x = np.ones((1, 1, 4, 4), dtype=np.float32)
    weight = np.ones((1, 1, 3, 3), dtype=np.float32)
    conv_mask_res = sp_pkg.sparse_conv2d_mask(x, weight, mask)
    assert conv_mask_res.nnz == 2
    np.testing.assert_allclose(conv_mask_res.values, [9.0, 9.0])

    # Eager dispatch
    res_eager_spmm = sp_pkg.execute_op(SparseGenerator, "spmm", csr, dense_rhs)
    np.testing.assert_allclose(res_eager_spmm, expected_prod)
    res_eager_spgemm = sp_pkg.execute_op(SparseGenerator, "spgemm", csr, csr_b)
    np.testing.assert_allclose(res_eager_spgemm.to_dense(), expected_spgemm)
