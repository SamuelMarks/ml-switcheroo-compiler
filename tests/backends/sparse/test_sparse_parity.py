"""Exhaustive parity, kernel routing, and profiling validation test suite for Sparse backend."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.sparse import kernels
from ml_switcheroo_compiler.backends.sparse.eager import _dispatch_sparse_linalg, execute_op
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.sparse.profiler import SparseProfiler, _get_process_memory_mb
from ml_switcheroo_compiler.backends.sparse.types import (
    COOTensor,
    CSCTensor,
    CSRTensor,
)
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def _make_test_adj(n: int = 4) -> tuple[np.ndarray, COOTensor]:
    """Create a sample graph adjacency matrix and corresponding COOTensor.

    Args:
        n (int): Number of nodes.

    Returns:
        tuple[np.ndarray, COOTensor]: Dense matrix and sparse COO tensor.
    """
    dense_adj = np.zeros((n, n), dtype=np.float32)
    dense_adj[0, 1] = 1.0
    dense_adj[1, 0] = 1.0
    dense_adj[1, 2] = 2.0
    dense_adj[2, 1] = 2.0
    dense_adj[2, 3] = 1.5
    dense_adj[3, 2] = 1.5
    dense_adj[0, 3] = 0.5
    dense_adj[3, 0] = 0.5
    return dense_adj, COOTensor.from_dense(dense_adj)


def test_sparse_dense_and_sparse_sparse_linalg_parity() -> None:
    """Verify numerical parity of sparse-dense (SpMM) and sparse-sparse (SpGEMM) operations."""
    dense_a, coo_a = _make_test_adj(4)
    csr_a = coo_a.to_csr()
    csc_a = coo_a.to_csc()

    dense_b = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=np.float32,
    )
    expected_spmm = dense_a @ dense_b

    # 1. SpMM: Sparse x Dense (COO, CSR, CSC)
    res_coo_dense = execute_op(SparseGenerator, "MatMul", coo_a, dense_b)
    np.testing.assert_allclose(res_coo_dense, expected_spmm, rtol=1e-5, atol=1e-5)

    res_csr_dense = execute_op(SparseGenerator, "matmul", csr_a, dense_b)
    np.testing.assert_allclose(res_csr_dense, expected_spmm, rtol=1e-5, atol=1e-5)

    res_csc_dense = execute_op(SparseGenerator, "Dot", csc_a, dense_b)
    np.testing.assert_allclose(res_csc_dense, expected_spmm, rtol=1e-5, atol=1e-5)

    # 2. Dense SpMM: Dense x Sparse (Dense x COO, Dense x CSR, Dense x CSC)
    dense_left = dense_b.T
    expected_dense_spmm = dense_left @ dense_a

    res_dense_coo = execute_op(SparseGenerator, "matmul", dense_left, coo_a)
    np.testing.assert_allclose(res_dense_coo, expected_dense_spmm, rtol=1e-5, atol=1e-5)

    res_dense_csr = execute_op(SparseGenerator, "Dot", dense_left, csr_a)
    np.testing.assert_allclose(res_dense_csr, expected_dense_spmm, rtol=1e-5, atol=1e-5)

    res_dense_csc = execute_op(SparseGenerator, "MatMul", dense_left, csc_a)
    np.testing.assert_allclose(res_dense_csc, expected_dense_spmm, rtol=1e-5, atol=1e-5)

    # 3. SpGEMM: Sparse x Sparse (COO x COO, CSR x CSR, CSC x CSC)
    expected_spgemm = dense_a @ dense_a

    res_spgemm_coo = execute_op(SparseGenerator, "MatMul", coo_a, coo_a)
    assert isinstance(res_spgemm_coo, COOTensor)
    np.testing.assert_allclose(res_spgemm_coo.to_dense(), expected_spgemm, rtol=1e-5, atol=1e-5)

    res_spgemm_csr = execute_op(SparseGenerator, "matmul", csr_a, csr_a)
    assert isinstance(res_spgemm_csr, CSRTensor)
    np.testing.assert_allclose(res_spgemm_csr.to_dense(), expected_spgemm, rtol=1e-5, atol=1e-5)

    res_spgemm_csc = execute_op(SparseGenerator, "Dot", csc_a, csc_a)
    assert isinstance(res_spgemm_csc, CSCTensor)
    np.testing.assert_allclose(res_spgemm_csc.to_dense(), expected_spgemm, rtol=1e-5, atol=1e-5)


def test_csr_and_csc_arithmetic() -> None:
    """Verify addition and subtraction across CSR and CSC formats."""
    dense_1, coo_1 = _make_test_adj(4)
    dense_2 = np.ones((4, 4), dtype=np.float32)
    coo_2 = COOTensor.from_dense(dense_2)

    csr_1, csr_2 = coo_1.to_csr(), coo_2.to_csr()
    csc_1, csc_2 = coo_1.to_csc(), coo_2.to_csc()

    # CSR Add and Sub
    res_csr_add = execute_op(SparseGenerator, "Add", csr_1, csr_2)
    np.testing.assert_allclose(res_csr_add.to_dense(), dense_1 + dense_2, rtol=1e-5, atol=1e-5)

    res_csr_sub = execute_op(SparseGenerator, "Sub", csr_1, csr_2)
    np.testing.assert_allclose(res_csr_sub.to_dense(), dense_1 - dense_2, rtol=1e-5, atol=1e-5)

    # CSC Add and Sub
    res_csc_add = execute_op(SparseGenerator, "add", csc_1, csc_2)
    np.testing.assert_allclose(res_csc_add.to_dense(), dense_1 + dense_2, rtol=1e-5, atol=1e-5)

    res_csc_sub = execute_op(SparseGenerator, "sub", csc_1, csc_2)
    np.testing.assert_allclose(res_csc_sub.to_dense(), dense_1 - dense_2, rtol=1e-5, atol=1e-5)

    # Empty matrix edge cases
    empty_coo = COOTensor(indices=np.zeros((2, 0), dtype=np.int64), values=np.zeros(0, dtype=np.float32), shape=(4, 4))
    empty_csr = empty_coo.to_csr()
    empty_csc = empty_coo.to_csc()

    assert kernels.csr_add(empty_csr, csr_1).to_dense().shape == (4, 4)
    assert kernels.csr_add(csr_1, empty_csr).to_dense().shape == (4, 4)
    assert kernels.csr_sub(empty_csr, csr_1).to_dense().shape == (4, 4)
    assert kernels.csr_sub(csr_1, empty_csr).to_dense().shape == (4, 4)

    assert kernels.csc_add(empty_csc, csc_1).to_dense().shape == (4, 4)
    assert kernels.csc_add(csc_1, empty_csc).to_dense().shape == (4, 4)
    assert kernels.csc_sub(empty_csc, csc_1).to_dense().shape == (4, 4)
    assert kernels.csc_sub(csc_1, empty_csc).to_dense().shape == (4, 4)


def test_gnn_attention_and_adjacency_kernels() -> None:
    """Verify custom sparse kernels for GNN attention and graph adjacency matrices."""
    dense_adj, coo_adj = _make_test_adj(4)

    # 1. sparse_softmax
    attn_coo = kernels.sparse_softmax(coo_adj)
    assert isinstance(attn_coo, COOTensor)
    attn_dense = attn_coo.to_dense()

    # Verify rows with edges sum to 1.0
    for r in range(4):
        if np.any(dense_adj[r] != 0):
            assert np.isclose(np.sum(attn_dense[r]), 1.0, atol=1e-5)
        else:
            assert np.isclose(np.sum(attn_dense[r]), 0.0)

    # Empty tensor edge case for sparse_softmax
    empty_coo = COOTensor(indices=np.zeros((2, 0), dtype=np.int64), values=np.zeros(0, dtype=np.float32), shape=(4, 4))
    assert kernels.sparse_softmax(empty_coo).nnz == 0

    # With structural mask
    masked_softmax = kernels.sparse_softmax(coo_adj, mask=coo_adj)
    assert masked_softmax.nnz == coo_adj.nnz

    # 2. graph_norm_adjacency
    norm_adj = kernels.graph_norm_adjacency(coo_adj, add_self_loops=True)
    assert isinstance(norm_adj, COOTensor)
    assert norm_adj.shape == (4, 4)

    # Numerical check: D^{-1/2} A_hat D^{-1/2}
    a_hat = dense_adj + np.eye(4, dtype=np.float32)
    deg_hat = np.sum(a_hat, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg_hat))
    expected_norm_adj = deg_inv_sqrt @ a_hat @ deg_inv_sqrt
    np.testing.assert_allclose(norm_adj.to_dense(), expected_norm_adj, rtol=1e-5, atol=1e-5)

    # Without self loops
    norm_adj_no_loops = kernels.graph_norm_adjacency(coo_adj, add_self_loops=False)
    assert norm_adj_no_loops.shape == (4, 4)

    # 3. gnn_spmm_attention
    node_features = np.array(
        [
            [1.0, 0.5],
            [0.2, 0.8],
            [1.5, 0.1],
            [0.7, 0.9],
        ],
        dtype=np.float32,
    )
    # Aggregation without custom weights
    agg_out = kernels.gnn_spmm_attention(coo_adj, node_features)
    assert agg_out.shape == (4, 2)
    np.testing.assert_allclose(agg_out, dense_adj @ node_features, rtol=1e-5, atol=1e-5)

    # Aggregation with attention weights
    agg_attn_out = kernels.gnn_spmm_attention(coo_adj, node_features, attention_weights=coo_adj)
    assert agg_attn_out.shape == (4, 2)

    # 4. sparse_dropout
    dropped = kernels.sparse_dropout(coo_adj, drop_rate=0.5, training=True, seed=42)
    assert isinstance(dropped, COOTensor)
    assert dropped.shape == coo_adj.shape

    # Not training mode
    no_drop = kernels.sparse_dropout(coo_adj, drop_rate=0.5, training=False)
    assert no_drop.nnz == coo_adj.nnz

    # Zero drop rate
    zero_drop = kernels.sparse_dropout(coo_adj, drop_rate=0.0, training=True)
    assert zero_drop.nnz == coo_adj.nnz

    # 100% drop rate
    full_drop = kernels.sparse_dropout(coo_adj, drop_rate=1.0, training=True)
    assert full_drop.nnz == 0

    # CSR and CSC dropout
    csr_drop = kernels.sparse_dropout(coo_adj.to_csr(), drop_rate=0.5, training=True, seed=42)
    assert isinstance(csr_drop, CSRTensor)

    csc_drop = kernels.sparse_dropout(coo_adj.to_csc(), drop_rate=0.5, training=True, seed=42)
    assert isinstance(csc_drop, CSCTensor)

    empty_drop = kernels.sparse_dropout(empty_coo, drop_rate=0.5, training=True)
    assert empty_drop.nnz == 0


def test_sparse_generator_visitors_and_compilation() -> None:
    """Verify SparseGenerator memory-efficient emission, dedicated visitors, and execution."""
    g = IRGraph()
    gen = SparseGenerator(g, memory_efficient=True)
    assert gen.memory_efficient is True
    code = gen.generate()
    assert "# Memory-efficient sparse execution active" in code
    assert "import numpy as np" in code

    gen_non_eff = SparseGenerator(g, memory_efficient=False)
    assert "# Memory-efficient sparse execution active" not in gen_non_eff.generate()

    # Dedicated visitors
    node_mock = IRNode(id="test_node", op_type="Spmm")
    assert gen.visit_Spmm(node_mock, ["a", "b"]) == "sp_kernels.spmm(a, b)"
    assert gen.visit_Spgemm(node_mock, ["a", "b"]) == "sp_kernels.spgemm(a, b)"
    assert gen.visit_DenseSpmm(node_mock, ["a", "b"]) == "sp_kernels.dense_spmm(a, b)"
    assert gen.visit_SparseMask(node_mock, ["a", "b"]) == "sp_kernels.sparse_mask(a, b)"
    assert gen.visit_SparseSoftmax(node_mock, ["a"]) == "sp_kernels.sparse_softmax(a)"
    assert gen.visit_GraphNormAdjacency(node_mock, ["a"]) == "sp_kernels.graph_norm_adjacency(a)"
    assert gen.visit_GnnSpmmAttention(node_mock, ["a", "b"]) == "sp_kernels.gnn_spmm_attention(a, b)"
    assert gen.visit_SparseDropout(node_mock, ["a"]) == "sp_kernels.sparse_dropout(a)"
    assert gen.visit_CsrAdd(node_mock, ["a", "b"]) == "sp_kernels.csr_add(a, b)"
    assert gen.visit_CscAdd(node_mock, ["a", "b"]) == "sp_kernels.csc_add(a, b)"
    assert gen.visit_CsrSub(node_mock, ["a", "b"]) == "sp_kernels.csr_sub(a, b)"
    assert gen.visit_CscSub(node_mock, ["a", "b"]) == "sp_kernels.csc_sub(a, b)"
    assert gen.visit_Add(node_mock, ["a", "b"]) == "sp_kernels.coo_add(a, b)"
    assert gen.visit_Sub(node_mock, ["a", "b"]) == "sp_kernels.coo_sub(a, b)"
    assert gen.visit_Mul(node_mock, ["a", "b"]) == "sp_kernels.coo_mul(a, b)"
    assert gen.visit_MatMul(node_mock, ["a", "b"]) == "sp_kernels.coo_matmat(a, b)"


def test_sparse_profiler_and_memory_metrics() -> None:
    """Verify execution profiling, latencies, percentiles, and peak memory tracking."""
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_sum = IRNode(id="out", op_type="Sum", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_in, "out": n_sum}
    g.inputs = ["x"]
    g.outputs = ["out"]

    profiler = SparseProfiler()
    metrics = profiler.profile_graph(
        g,
        inputs={"x": [[1.0, 2.0], [3.0, 4.0]]},
        num_iters=3,
        warmup_iters=1,
    )

    assert "latencies" in metrics
    assert len(metrics["latencies"]) == 3
    assert metrics["latency_ms"] >= 0.0
    assert metrics["p50_ms"] >= 0.0
    assert metrics["p95_ms"] >= 0.0
    assert metrics["p99_ms"] >= 0.0
    assert metrics["peak_memory_mb"] > 0.0

    # Test Linux RSS branch in _get_process_memory_mb
    with mock.patch("sys.platform", "linux"):
        assert _get_process_memory_mb() > 0.0

    # Profile with empty inputs
    empty_graph = IRGraph()
    metrics_empty = profiler.profile_graph(empty_graph, inputs={}, num_iters=2, warmup_iters=1)
    assert len(metrics_empty["latencies"]) == 2

    # Profile with fallback summation step
    class MockAddTensor:
        def __add__(self, other: object) -> object:
            return self

    class MockNonAddTensor:
        pass

    metrics_add = profiler.profile_graph(empty_graph, inputs={"a": MockAddTensor(), "b": MockAddTensor()}, num_iters=2)
    assert len(metrics_add["latencies"]) == 2

    metrics_non_add = profiler.profile_graph(empty_graph, inputs={"a": MockNonAddTensor(), "b": MockNonAddTensor()}, num_iters=2)
    assert len(metrics_non_add["latencies"]) == 2


def test_sparse_eager_dispatch_coverage() -> None:
    """Verify eager dispatch error cases and unmapped operations."""
    dense_adj, coo_adj = _make_test_adj(4)

    # Direct GNN and graph ops in execute_op
    soft_res = execute_op(SparseGenerator, "sparse_softmax", coo_adj)
    assert isinstance(soft_res, COOTensor)

    norm_res = execute_op(SparseGenerator, "graph_norm_adjacency", coo_adj)
    assert isinstance(norm_res, COOTensor)

    drop_res = execute_op(SparseGenerator, "sparse_dropout", coo_adj)
    assert isinstance(drop_res, COOTensor)

    # Unhandled linalg dispatch fallback
    handled, _ = _dispatch_sparse_linalg("unknown_op", coo_adj, coo_adj)
    assert handled is False

    # Unsupported operation
    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(SparseGenerator, "NonexistentUnknownSparseOp", coo_adj)


def test_sparse_edge_cases_and_branch_coverage() -> None:
    """Verify edge conditions in eager linalg dispatch, kernels, and profiler."""
    dense_adj, coo_adj = _make_test_adj(4)
    csr_adj = coo_adj.to_csr()
    csc_adj = coo_adj.to_csc()

    # 1. _dispatch_sparse_linalg: non-sparse operands for Add and Sub, and mixed CSR/CSC
    handled_dense_matmul, _ = _dispatch_sparse_linalg("MatMul", np.array([1]), np.array([2]))
    assert handled_dense_matmul is False

    handled_add, _ = _dispatch_sparse_linalg("Add", np.array([1]), np.array([2]))
    assert handled_add is False

    handled_mixed_add, _ = _dispatch_sparse_linalg("Add", csr_adj, csc_adj)
    assert handled_mixed_add is False

    handled_coo_add, _ = _dispatch_sparse_linalg("Add", coo_adj, coo_adj)
    assert handled_coo_add is False

    handled_sub, _ = _dispatch_sparse_linalg("Sub", np.array([1]), np.array([2]))
    assert handled_sub is False

    # 2. kernels: empty operand branches in csr_sub and csc_sub
    empty_coo = COOTensor(indices=np.zeros((2, 0), dtype=np.int64), values=np.zeros(0, dtype=np.float32), shape=(4, 4))
    empty_csr = empty_coo.to_csr()
    empty_csc = empty_coo.to_csc()

    with pytest.raises(ValueError, match="Shape mismatch"):
        kernels.csr_sub(csr_adj, CSRTensor.from_dense(np.ones((2, 2))))

    with pytest.raises(ValueError, match="Shape mismatch"):
        kernels.csc_sub(csc_adj, CSCTensor.from_dense(np.ones((2, 2))))

    sub_csr_res = kernels.csr_sub(empty_csr, csr_adj)
    assert isinstance(sub_csr_res, CSRTensor)
    np.testing.assert_allclose(sub_csr_res.to_dense(), -dense_adj)

    sub_csr_b_empty = kernels.csr_sub(csr_adj, empty_csr)
    assert sub_csr_b_empty is csr_adj

    sub_csc_res = kernels.csc_sub(empty_csc, csc_adj)
    assert isinstance(sub_csc_res, CSCTensor)
    np.testing.assert_allclose(sub_csc_res.to_dense(), -dense_adj)

    sub_csc_b_empty = kernels.csc_sub(csc_adj, empty_csc)
    assert sub_csc_b_empty is csc_adj

    # 3. sparse_dropout: drop_rate >= 1.0 on CSR and CSC
    csr_100 = kernels.sparse_dropout(csr_adj, drop_rate=1.0)
    assert isinstance(csr_100, CSRTensor)
    assert csr_100.nnz == 0

    csc_100 = kernels.sparse_dropout(csc_adj, drop_rate=1.0)
    assert isinstance(csc_100, CSCTensor)
    assert csc_100.nnz == 0

    # 4. sparse_dropout: not np.any(keep_mask) branch
    fake_rng = mock.MagicMock()
    fake_rng.uniform.return_value = np.ones(coo_adj.nnz)
    with mock.patch("numpy.random.default_rng", return_value=fake_rng):
        drop_all = kernels.sparse_dropout(coo_adj, drop_rate=0.5)
        assert drop_all.nnz == 0

    # 5. profiler: inputs with COOTensor, np.ndarray, list, and other types
    profiler = SparseProfiler()
    g = IRGraph()
    prepared = profiler._prepare_inputs(
        g,
        {
            "coo": coo_adj,
            "arr": np.array([1.0, 2.0]),
            "lst": [3.0, 4.0],
            "raw": 42,
        },
    )
    assert len(prepared) == 4

    try:
        import sparse as pydata_sparse

        pydata_coo = pydata_sparse.COO.from_numpy(np.array([1.0, 2.0]))
        prepared_pydata = profiler._prepare_inputs(g, {"pydata": pydata_coo})
        assert len(prepared_pydata) == 1
    except ImportError:
        pass

    # 6. profiler: _compile_graph failure fallback
    g_sum = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_out = IRNode(id="out", op_type="Sum", inputs=["x"], shape_metadata=[2, 2])
    g_sum.nodes = {"x": n_in, "out": n_out}
    g_sum.inputs = ["x"]
    g_sum.outputs = ["out"]

    with mock.patch.object(profiler, "_compile_graph", side_effect=RuntimeError("compile failed")):
        metrics_fail = profiler.profile_graph(g_sum, inputs={"x": [[1.0, 2.0], [3.0, 4.0]]}, num_iters=2)
        assert len(metrics_fail["latencies"]) == 2

    # 7. profiler import fallback when sparse library is not present
    real_import = __import__

    def fake_sparse_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sparse":
            raise ImportError("no sparse module")
        return real_import(name, *args, **kwargs)

    import importlib

    with mock.patch("builtins.__import__", side_effect=fake_sparse_import):
        import ml_switcheroo_compiler.backends.sparse.profiler as sp_prof_mod

        importlib.reload(sp_prof_mod)

    # Restore profiler module
    import ml_switcheroo_compiler.backends.sparse.profiler as sp_prof_mod

    importlib.reload(sp_prof_mod)
