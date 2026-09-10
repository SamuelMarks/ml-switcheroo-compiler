"""Comprehensive tests for Sparse COO backend, kernels, generator, and eager execution."""

import importlib
import sys
from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.backends.sparse.eager import execute_op
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.sparse.kernels import (
    coo_abs,
    coo_add,
    coo_dot,
    coo_fromdense,
    coo_matmat,
    coo_matvec,
    coo_mean,
    coo_mul,
    coo_neg,
    coo_relu,
    coo_reshape,
    coo_sub,
    coo_sum,
    coo_sum_duplicates,
    coo_todense,
    coo_transpose,
)
from ml_switcheroo_compiler.backends.sparse.types import COOTensor, array, asarray, item, zeros
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class DummyGraph:
    """Dummy graph for generator testing."""

    def __init__(self) -> None:
        """Initialize empty dummy graph."""
        self.nodes: list[IRNode] = []
        self.inputs: list[str] = ["x"]
        self.outputs: list[str] = ["out"]


def test_sparse_backend_registered() -> None:
    """Verify registration in BackendRegistry for sparse and sparse_coo."""
    assert BackendRegistry.get("sparse") is SparseGenerator
    assert BackendRegistry.get("sparse_coo") is SparseGenerator


def test_coo_tensor_properties_and_methods() -> None:
    """Test COOTensor initialization, properties, dense roundtrip, and types."""
    dense_2d = np.array([[0.0, 1.5, 0.0], [2.0, 0.0, 3.5]], dtype=np.float32)
    coo = COOTensor.from_dense(dense_2d)

    assert coo.shape == (2, 3)
    assert coo.ndim == 2
    assert coo.nnz == 3
    assert coo.size == 6
    assert coo.dtype == np.float32
    assert "<COOTensor shape=(2, 3) nnz=3" in repr(coo)

    reconstructed = coo.to_dense()
    np.testing.assert_allclose(reconstructed, dense_2d)

    # astype
    coo_f64 = coo.astype(np.float64)
    assert coo_f64.dtype == np.float64
    np.testing.assert_allclose(coo_f64.to_dense(), dense_2d)

    # 1D COOTensor
    dense_1d = np.array([0.0, 4.0, 0.0, 5.0])
    coo_1d = COOTensor.from_dense(dense_1d)
    assert coo_1d.shape == (4,)
    assert coo_1d.ndim == 1
    assert coo_1d.indices.shape == (1, 2)
    np.testing.assert_allclose(coo_1d.to_dense(), dense_1d)

    # 1D explicit reshape
    coo_1d_direct = COOTensor(indices=np.array([1, 3]), values=np.array([4.0, 5.0]), shape=(4,))
    assert coo_1d_direct.indices.shape == (1, 2)

    # Float indices conversion to int64
    coo_flt = COOTensor(indices=np.array([[0.0, 1.0], [1.0, 0.0]]), values=np.array([1.0, 2.0]), shape=(2, 2))
    assert coo_flt.indices.dtype == np.int64

    # 1D indices for multi-dimensional shape
    coo_1d_nd = COOTensor(indices=np.array([0, 1]), values=np.array([3.0]), shape=(2, 1))
    assert coo_1d_nd.indices.shape == (2, 1)


def test_sparse_types_helpers() -> None:
    """Test zeros, array, asarray, and item helpers."""
    z = zeros(type, (3, 4))
    assert isinstance(z, COOTensor)
    assert z.shape == (3, 4)
    assert z.nnz == 0
    np.testing.assert_allclose(z.to_dense(), np.zeros((3, 4)))

    arr = array(type, [0.0, 2.5, 0.0])
    assert isinstance(arr, COOTensor)
    assert arr.nnz == 1

    arr_dtype = array(type, [1.0, 2.0], dtype="float64")
    assert arr_dtype.dtype == np.float64

    coo_identity = array(type, arr)
    assert coo_identity is arr

    coo_converted = array(type, arr, dtype="float64")
    assert coo_converted.dtype == np.float64

    as_arr = asarray(type, [1.0, 0.0])
    assert isinstance(as_arr, COOTensor)
    assert asarray(type, as_arr) is as_arr

    scalar_coo = COOTensor.from_dense(np.array(7.5))
    assert item(type, scalar_coo) == 7.5
    assert item(type, np.array(8.5)) == 8.5


def test_coo_sum_duplicates() -> None:
    """Test duplicate coordinate merging and zero cancellation."""
    indices = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
    values = np.array([2.0, 3.0, 4.0, -4.0])
    shape = (2, 3)

    canonical = coo_sum_duplicates(indices, values, shape)
    assert canonical.nnz == 1
    np.testing.assert_array_equal(canonical.indices, np.array([[0], [1]]))
    np.testing.assert_array_equal(canonical.values, np.array([5.0]))

    # Empty values
    empty = coo_sum_duplicates(np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32), (2, 2))
    assert empty.nnz == 0

    # 1D indices
    indices_1d = np.array([0, 0, 1])
    values_1d = np.array([1.0, 2.0, 3.0])
    canonical_1d = coo_sum_duplicates(indices_1d, values_1d, (2,))
    assert canonical_1d.nnz == 2
    np.testing.assert_allclose(canonical_1d.to_dense(), [3.0, 3.0])

    # All cancelling to zero
    zero_vals = np.array([1.0, -1.0])
    zero_idx = np.array([[0, 0], [0, 0]])
    zero_coo = coo_sum_duplicates(zero_idx, zero_vals, (1, 1))
    assert zero_coo.nnz == 0


def test_coo_fromdense_and_todense() -> None:
    """Test coo_fromdense and coo_todense conversions against NumPy reference."""
    dense = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    coo = coo_fromdense(dense, index_dtype=np.int32)
    assert coo.indices.dtype == np.int32
    assert coo_todense(coo) is not None
    np.testing.assert_allclose(coo_todense(coo), dense)
    np.testing.assert_allclose(coo_todense(dense), dense)

    # Pass existing COOTensor
    coo_id = coo_fromdense(coo)
    assert coo_id is coo
    coo_cast_idx = coo_fromdense(coo, index_dtype=np.int64)
    assert coo_cast_idx.indices.dtype == np.int64


def test_coo_matmat_dense_and_sparse_parity() -> None:
    """Verify coo_matmat against dense NumPy @ reference."""
    rng = np.random.default_rng(42)
    dense_a = rng.choice([0.0, 1.0, 2.0], size=(4, 5), p=[0.7, 0.2, 0.1])
    dense_b = rng.normal(size=(5, 3))

    coo_a = coo_fromdense(dense_a)

    # Sparse @ Dense
    prod_dense = coo_matmat(coo_a, dense_b)
    expected_dense = dense_a @ dense_b
    np.testing.assert_allclose(prod_dense, expected_dense, rtol=1e-5, atol=1e-5)

    # Transposed Sparse @ Dense
    dense_a_t = dense_a.T
    prod_trans = coo_matmat(coo_a, rng.normal(size=(4, 3)), transpose=True)
    expected_trans = dense_a_t @ rng.normal(size=(4, 3))
    # Test with consistent matrix
    mat_right = rng.normal(size=(4, 2))
    prod_trans = coo_matmat(coo_a, mat_right, transpose=True)
    expected_trans = dense_a.T @ mat_right
    np.testing.assert_allclose(prod_trans, expected_trans, rtol=1e-5, atol=1e-5)

    # Sparse @ Sparse
    dense_b_sparse = rng.choice([0.0, 1.0, -1.0], size=(5, 6), p=[0.8, 0.1, 0.1])
    coo_b = coo_fromdense(dense_b_sparse)
    prod_sp = coo_matmat(coo_a, coo_b)
    assert isinstance(prod_sp, COOTensor)
    expected_sp = dense_a @ dense_b_sparse
    np.testing.assert_allclose(prod_sp.to_dense(), expected_sp, rtol=1e-5, atol=1e-5)

    # Sparse @ Sparse with empty / disjoint matrices
    empty_coo = zeros(type, (4, 5))
    prod_empty = coo_matmat(empty_coo, coo_b)
    assert isinstance(prod_empty, COOTensor)
    assert prod_empty.nnz == 0

    disjoint_a = COOTensor(indices=np.array([[0], [0]]), values=np.array([1.0]), shape=(2, 2))
    disjoint_b = COOTensor(indices=np.array([[1], [0]]), values=np.array([1.0]), shape=(2, 2))
    prod_disjoint = coo_matmat(disjoint_a, disjoint_b)
    assert isinstance(prod_disjoint, COOTensor)
    assert prod_disjoint.nnz == 0

    # Empty matrix with dense matrix
    empty_m = zeros(type, (4, 5))
    prod_empty_dense = coo_matmat(empty_m, dense_b)
    np.testing.assert_allclose(prod_empty_dense, np.zeros((4, 3)))


def test_coo_matvec_parity() -> None:
    """Verify coo_matvec against dense NumPy matrix-vector multiplication."""
    dense_m = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    v = np.array([2.0, 3.0, 4.0])

    coo_m = coo_fromdense(dense_m)

    res = coo_matvec(coo_m, v)
    expected = dense_m @ v
    np.testing.assert_allclose(res, expected)

    # Transposed
    res_t = coo_matvec(coo_m, v, transpose=True)
    expected_t = dense_m.T @ v
    np.testing.assert_allclose(res_t, expected_t)

    # Sparse vector operand
    v_sparse = coo_fromdense(v)
    res_sp_v = coo_matvec(coo_m, v_sparse)
    np.testing.assert_allclose(res_sp_v, expected)

    # Empty matrix @ vector
    empty_mat = zeros(type, (3, 3))
    res_empty = coo_matvec(empty_mat, v)
    np.testing.assert_allclose(res_empty, np.zeros(3))


def test_coo_transpose_parity() -> None:
    """Verify coo_transpose against dense NumPy transpose."""
    dense_3d = np.arange(24).reshape(2, 3, 4)
    coo_3d = coo_fromdense(dense_3d)

    # Default reverse transpose
    trans_def = coo_transpose(coo_3d)
    np.testing.assert_allclose(trans_def.to_dense(), dense_3d.transpose())

    # Explicit axes
    trans_perm = coo_transpose(coo_3d, axes=[1, 2, 0])
    np.testing.assert_allclose(trans_perm.to_dense(), dense_3d.transpose(1, 2, 0))

    # Empty tensor transpose
    empty_coo = zeros(type, (2, 3))
    empty_trans = coo_transpose(empty_coo)
    assert empty_trans.shape == (3, 2)
    assert empty_trans.nnz == 0


def test_coo_add_sub_parity() -> None:
    """Verify coo_add and coo_sub against dense NumPy addition and subtraction."""
    a = np.array([[1.0, 0.0], [2.0, 3.0]])
    b = np.array([[0.0, 4.0], [5.0, 0.0]])

    coo_a = coo_fromdense(a)
    coo_b = coo_fromdense(b)

    # Sparse + Sparse
    sum_ab = coo_add(coo_a, coo_b)
    assert isinstance(sum_ab, COOTensor)
    np.testing.assert_allclose(sum_ab.to_dense(), a + b)

    # Sparse - Sparse
    diff_ab = coo_sub(coo_a, coo_b)
    assert isinstance(diff_ab, COOTensor)
    np.testing.assert_allclose(diff_ab.to_dense(), a - b)

    # Identity with empty
    empty = zeros(type, (2, 2))
    assert coo_add(coo_a, empty) is coo_a
    assert coo_add(empty, coo_a) is coo_a

    # Mixed Sparse + Dense
    dense_c = np.ones((2, 2))
    np.testing.assert_allclose(coo_add(coo_a, dense_c), a + dense_c)
    np.testing.assert_allclose(coo_add(dense_c, coo_a), dense_c + a)
    np.testing.assert_allclose(coo_sub(coo_a, dense_c), a - dense_c)
    assert coo_add(1.0, 2.0) == 3.0

    # Shape mismatch error
    with pytest.raises(ValueError, match="Shape mismatch"):
        coo_add(coo_a, zeros(type, (3, 3)))


def test_coo_mul_parity() -> None:
    """Verify coo_mul against dense NumPy element-wise multiplication."""
    a = np.array([[2.0, 0.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [0.0, 7.0]])

    coo_a = coo_fromdense(a)
    coo_b = coo_fromdense(b)

    # Sparse * Sparse
    prod_sp = coo_mul(coo_a, coo_b)
    assert isinstance(prod_sp, COOTensor)
    np.testing.assert_allclose(prod_sp.to_dense(), a * b)

    # Disjoint sparse multiply
    d1 = COOTensor(indices=np.array([[0], [0]]), values=np.array([1.0]), shape=(2, 2))
    d2 = COOTensor(indices=np.array([[1], [1]]), values=np.array([2.0]), shape=(2, 2))
    prod_disjoint = coo_mul(d1, d2)
    assert isinstance(prod_disjoint, COOTensor)
    assert prod_disjoint.nnz == 0

    # Sparse * Scalar (non-zero and zero)
    prod_sc = coo_mul(coo_a, 3.5)
    assert isinstance(prod_sc, COOTensor)
    np.testing.assert_allclose(prod_sc.to_dense(), a * 3.5)

    prod_sc_0 = coo_mul(coo_a, 0.0)
    assert isinstance(prod_sc_0, COOTensor)
    assert prod_sc_0.nnz == 0

    prod_sc_left = coo_mul(2.0, coo_a)
    assert isinstance(prod_sc_left, COOTensor)
    np.testing.assert_allclose(prod_sc_left.to_dense(), 2.0 * a)

    # Sparse * Dense
    dense_d = np.array([[10.0, 20.0], [30.0, 40.0]])
    prod_dense = coo_mul(coo_a, dense_d)
    assert isinstance(prod_dense, COOTensor)
    np.testing.assert_allclose(prod_dense.to_dense(), a * dense_d)

    prod_dense_rev = coo_mul(dense_d, coo_a)
    assert isinstance(prod_dense_rev, COOTensor)
    np.testing.assert_allclose(prod_dense_rev.to_dense(), dense_d * a)

    # Empty multiply
    empty = zeros(type, (2, 2))
    assert coo_mul(empty, coo_b).nnz == 0
    assert coo_mul(empty, dense_d).nnz == 0
    np.testing.assert_allclose(coo_mul(np.array([2.0]), np.array([3.0])), [6.0])

    # Shape mismatch error
    with pytest.raises(ValueError, match="Shape mismatch"):
        coo_mul(coo_a, zeros(type, (3, 3)))


def test_coo_neg_abs_relu_parity() -> None:
    """Verify coo_neg, coo_abs, and coo_relu against dense NumPy reference."""
    dense = np.array([[-3.0, 0.0], [4.0, -5.0]])
    coo = coo_fromdense(dense)

    # Neg
    neg_res = coo_neg(coo)
    np.testing.assert_allclose(neg_res.to_dense(), -dense)

    # Abs
    abs_res = coo_abs(coo)
    np.testing.assert_allclose(abs_res.to_dense(), np.abs(dense))

    # ReLU
    relu_res = coo_relu(coo)
    np.testing.assert_allclose(relu_res.to_dense(), np.maximum(dense, 0.0))


def test_coo_reshape_parity() -> None:
    """Verify coo_reshape against dense NumPy reshape."""
    dense = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]])
    coo = coo_fromdense(dense)

    reshaped = coo_reshape(coo, (3, 2))
    assert reshaped.shape == (3, 2)
    np.testing.assert_allclose(reshaped.to_dense(), dense.reshape(3, 2))

    reshaped_1d = coo_reshape(coo, (6,))
    assert reshaped_1d.shape == (6,)
    np.testing.assert_allclose(reshaped_1d.to_dense(), dense.reshape(6))

    # Empty reshape
    empty = zeros(type, (2, 3))
    empty_reshaped = coo_reshape(empty, (6,))
    assert empty_reshaped.shape == (6,)
    assert empty_reshaped.nnz == 0


def test_coo_sum_and_mean_parity() -> None:
    """Verify coo_sum and coo_mean against dense NumPy reductions."""
    dense = np.array([[1.0, 0.0, 3.0], [0.0, 4.0, 5.0]])
    coo = coo_fromdense(dense)

    # Global sum
    s_glob = coo_sum(coo)
    assert s_glob == np.sum(dense)

    s_glob_kd = coo_sum(coo, keepdims=True)
    assert isinstance(s_glob_kd, COOTensor)
    assert s_glob_kd.shape == (1, 1)
    np.testing.assert_allclose(s_glob_kd.to_dense(), np.sum(dense, keepdims=True))

    # Axis sum
    s_ax0 = coo_sum(coo, axis=0)
    assert isinstance(s_ax0, COOTensor)
    np.testing.assert_allclose(s_ax0.to_dense(), np.sum(dense, axis=0))

    s_ax1_kd = coo_sum(coo, axis=1, keepdims=True)
    assert isinstance(s_ax1_kd, COOTensor)
    np.testing.assert_allclose(s_ax1_kd.to_dense(), np.sum(dense, axis=1, keepdims=True))

    # All axes specified as sequence
    s_all_axes = coo_sum(coo, axis=[0, 1])
    assert s_all_axes == np.sum(dense)

    # Global mean
    m_glob = coo_mean(coo)
    assert m_glob == np.mean(dense)

    m_glob_kd = coo_mean(coo, keepdims=True)
    assert isinstance(m_glob_kd, COOTensor)
    np.testing.assert_allclose(m_glob_kd.to_dense(), np.mean(dense, keepdims=True))

    # Axis mean
    m_ax0 = coo_mean(coo, axis=0)
    assert isinstance(m_ax0, COOTensor)
    np.testing.assert_allclose(m_ax0.to_dense(), np.mean(dense, axis=0))

    m_axes = coo_mean(coo, axis=[0, 1])
    assert m_axes == float(np.mean(dense))

    # Empty sum and mean
    empty = zeros(type, (2, 3))
    assert coo_sum(empty) == 0.0
    assert coo_sum(empty, keepdims=True).shape == (1, 1)
    assert coo_sum(empty, axis=0).shape == (3,)
    assert coo_sum(empty, axis=0, keepdims=True).shape == (1, 3)
    assert coo_mean(empty) == 0.0


def test_coo_dot_parity() -> None:
    """Verify coo_dot against dense dot product implementations."""
    # 2D @ 2D
    a2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    b2 = np.array([[5.0, 6.0], [7.0, 8.0]])
    res22 = coo_dot(coo_fromdense(a2), b2)
    np.testing.assert_allclose(res22, a2 @ b2)

    # 2D @ 1D
    v = np.array([2.0, 3.0])
    res21 = coo_dot(coo_fromdense(a2), v)
    np.testing.assert_allclose(res21, a2 @ v)

    # 1D @ 1D
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    res11 = coo_dot(coo_fromdense(v1), coo_fromdense(v2))
    assert res11 == float(np.dot(v1, v2))
    res11_dense = coo_dot(v1, v2)
    assert res11_dense == float(np.dot(v1, v2))

    # Higher-dim dot
    h1 = COOTensor(indices=np.zeros((3, 1), dtype=np.int64), values=np.array([1.0]), shape=(2, 2, 2))
    h2 = COOTensor(indices=np.zeros((3, 1), dtype=np.int64), values=np.array([2.0]), shape=(2, 2, 2))
    assert coo_dot(h1, h2) is not None


def test_sparse_generator_lifecycle_and_emission() -> None:
    """Test SparseGenerator source generation and execution pipeline."""
    graph = IRGraph()
    graph.nodes["in1"] = IRNode(id="in1", op_type="Input", inputs=[])
    graph.nodes["in2"] = IRNode(id="in2", op_type="Input", inputs=[])
    node = IRNode(id="out", op_type="Add", inputs=["in1", "in2"], attributes={})
    graph.nodes["out"] = node
    graph.inputs = ["in1", "in2"]
    graph.outputs = ["out"]

    gen = SparseGenerator(graph)
    assert gen.get_fallback_prefix() == "sp_kernels"
    assert gen.get_helper_functions() == []

    ops_map = gen.get_ops_map({})
    assert "coo_matmat" in ops_map
    assert "Add" in ops_map

    code = gen.generate()
    assert "from ml_switcheroo_compiler.backends.sparse import kernels as sp_kernels" in code
    assert "def evaluate(args):" in code

    fn = gen.compile_fn()
    assert callable(fn)

    dense_a = np.array([[1.0, 0.0], [0.0, 2.0]])
    dense_b = np.array([[0.0, 3.0], [4.0, 0.0]])
    coo_a = coo_fromdense(dense_a)
    coo_b = coo_fromdense(dense_b)

    res = fn([coo_a, coo_b])
    assert isinstance(res, COOTensor)
    np.testing.assert_allclose(res.to_dense(), dense_a + dense_b)

    # Generic visit
    node_unary = IRNode(id="u", op_type="Sin", inputs=["x"], attributes={})
    res_generic = gen.generic_visit(node_unary, ["x"])
    assert "sp_kernels.sin(x)" in res_generic

    # Compile error when callable missing
    with patch.object(gen, "generate", return_value="foo = 42"):
        with pytest.raises(RuntimeError, match="did not produce callable"):
            gen.compile_fn()


def test_sparse_eager_execute_op() -> None:
    """Test eager operation dispatch on Sparse backend."""
    a = coo_fromdense(np.array([[1.0, 0.0], [2.0, 0.0]]))
    b = np.array([[0.0, 3.0], [4.0, 0.0]])

    # Direct ops
    res_add = execute_op(type, "Add", a, b)
    np.testing.assert_allclose(res_add, a.to_dense() + b)

    res_matmat = execute_op(type, "coo_matmat", a, b)
    np.testing.assert_allclose(res_matmat, a.to_dense() @ b)

    res_sparse_coo_tensor = execute_op(type, "SparseCooTensor", np.array([[0], [1]]), np.array([3.0]), (2, 2))
    assert isinstance(res_sparse_coo_tensor, COOTensor)
    assert res_sparse_coo_tensor.nnz == 1

    # Schema mapping dispatch
    res_schema = execute_op(type, "coo_todense", a)
    np.testing.assert_allclose(res_schema, a.to_dense())

    with patch.dict("ml_switcheroo_compiler.backends.sparse.eager._DIRECT_OPS", clear=True):
        res_mapped = execute_op(type, "coo_todense", a)
        np.testing.assert_allclose(res_mapped, a.to_dense())

        with patch("ml_switcheroo_compiler.backends.mapping_loader.dispatch_eager_op", side_effect=BackendNotSupportedError("mock error")):
            global_eager_registry.register("coo_todense")(lambda mod, x: "dispatched_fallback")
            res_fb = execute_op(type, "coo_todense", a)
            assert res_fb == "dispatched_fallback"
            del global_eager_registry._registry["coo_todense"]

    # Global eager registry fallback with COOTensor arguments
    def custom_eager(backend_mod: object, x: object, scale: object = 2.0) -> np.ndarray:
        return np.asarray(x) * float(scale)

    global_eager_registry.register("TestSparseCustomOp")(custom_eager)
    res_custom = execute_op(type, "TestSparseCustomOp", a, 3.0)
    assert isinstance(res_custom, COOTensor)
    np.testing.assert_allclose(res_custom.to_dense(), a.to_dense() * 3.0)
    del global_eager_registry._registry["TestSparseCustomOp"]

    global_eager_registry.register("ScalarOp")(lambda mod, x: 42.0)
    res_scalar = execute_op(type, "ScalarOp", a)
    assert res_scalar == 42.0
    del global_eager_registry._registry["ScalarOp"]

    # BackendNotSupportedError
    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(type, "NonExistentSparseOpXYZ")


def test_sparse_init_import_guard() -> None:
    """Test import guard and package initialization."""
    import ml_switcheroo_compiler.backends.sparse as sp_pkg

    assert hasattr(sp_pkg, "SparseGenerator")
    assert hasattr(sp_pkg, "execute_op")
    assert hasattr(sp_pkg, "COOTensor")

    with patch("importlib.util.find_spec", side_effect=ValueError):
        importlib.reload(sp_pkg)

    with patch("importlib.util.find_spec", return_value=None):
        with patch.dict(sys.modules):
            if "pytest" in sys.modules:
                del sys.modules["pytest"]
            if "sphinx" in sys.modules:
                del sys.modules["sphinx"]
            with pytest.raises(ImportError, match="requires the 'sparse' library"):
                importlib.reload(sp_pkg)

    importlib.reload(sp_pkg)
