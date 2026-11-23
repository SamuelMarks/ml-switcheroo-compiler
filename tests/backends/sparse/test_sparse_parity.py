"""Exhaustive parity, kernel routing, and profiling validation test suite for Sparse backend."""

from __future__ import annotations

import builtins
import types
import typing
from collections.abc import Mapping, Sequence
from unittest import mock

import numpy as np
import pytest
from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph, NodeDict

from ml_switcheroo_compiler.backends.eager_registry import BackendArg
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.backends.sparse import kernels
from ml_switcheroo_compiler.backends.sparse.eager import _dispatch_sparse_linalg, execute_op
from ml_switcheroo_compiler.backends.sparse.generator import SparseGenerator
from ml_switcheroo_compiler.backends.sparse.profiler import SparseProfiler, _get_process_memory_mb
from ml_switcheroo_compiler.backends.sparse.types import COOTensor, CSCTensor, CSRTensor
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from tests.testing_utils import (
    EagerBoundCallable,
    MockTestStub,
    adapt_backend_module,
    coerce_operand_list,
    extract_test_shape,
    require_callable,
)


def _make_test_adj(n: int = 4) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.float32]], COOTensor]:
    """Create a symmetric sparse adjacency matrix and dense reference.

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
    return (dense_adj, COOTensor.from_dense(dense_adj))


def test_sparse_dense_and_sparse_sparse_linalg_parity() -> None:
    """Verify numerical parity of sparse-dense (SpMM) and sparse-sparse (SpGEMM) operations."""
    (dense_a, coo_a) = _make_test_adj(4)
    csr_a = coo_a.to_csr()
    csc_a = coo_a.to_csc()
    dense_b = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    expected_spmm = dense_a @ dense_b
    res_coo_dense = execute_op(SparseGenerator, "MatMul", coo_a, dense_b)
    np.testing.assert_allclose(ensure_ndarray(res_coo_dense), expected_spmm, rtol=1e-05, atol=1e-05)
    res_csr_dense = execute_op(SparseGenerator, "matmul", csr_a, dense_b)
    np.testing.assert_allclose(ensure_ndarray(res_csr_dense), expected_spmm, rtol=1e-05, atol=1e-05)
    res_csc_dense = execute_op(SparseGenerator, "Dot", csc_a, dense_b)
    np.testing.assert_allclose(ensure_ndarray(res_csc_dense), expected_spmm, rtol=1e-05, atol=1e-05)
    dense_left = dense_b.T
    expected_dense_spmm = dense_left @ dense_a
    res_dense_coo = execute_op(SparseGenerator, "matmul", dense_left, coo_a)
    np.testing.assert_allclose(ensure_ndarray(res_dense_coo), expected_dense_spmm, rtol=1e-05, atol=1e-05)
    res_dense_csr = execute_op(SparseGenerator, "Dot", dense_left, csr_a)
    np.testing.assert_allclose(ensure_ndarray(res_dense_csr), expected_dense_spmm, rtol=1e-05, atol=1e-05)
    res_dense_csc = execute_op(SparseGenerator, "MatMul", dense_left, csc_a)
    np.testing.assert_allclose(ensure_ndarray(res_dense_csc), expected_dense_spmm, rtol=1e-05, atol=1e-05)
    expected_spgemm = dense_a @ dense_a
    res_spgemm_coo = execute_op(SparseGenerator, "MatMul", coo_a, coo_a)
    assert isinstance(res_spgemm_coo, COOTensor)
    np.testing.assert_allclose(res_spgemm_coo.to_dense(), expected_spgemm, rtol=1e-05, atol=1e-05)
    res_spgemm_csr = execute_op(SparseGenerator, "matmul", csr_a, csr_a)
    assert isinstance(res_spgemm_csr, CSRTensor)
    np.testing.assert_allclose(res_spgemm_csr.to_dense(), expected_spgemm, rtol=1e-05, atol=1e-05)
    res_spgemm_csc = execute_op(SparseGenerator, "Dot", csc_a, csc_a)
    assert isinstance(res_spgemm_csc, CSCTensor)
    np.testing.assert_allclose(res_spgemm_csc.to_dense(), expected_spgemm, rtol=1e-05, atol=1e-05)


def test_csr_and_csc_arithmetic() -> None:
    """Verify addition and subtraction across CSR and CSC formats."""
    (dense_1, coo_1) = _make_test_adj(4)
    dense_2 = np.ones((4, 4), dtype=np.float32)
    coo_2 = COOTensor.from_dense(dense_2)
    (csr_1, csr_2) = (coo_1.to_csr(), coo_2.to_csr())
    (csc_1, csc_2) = (coo_1.to_csc(), coo_2.to_csc())
    res_csr_add = execute_op(SparseGenerator, "Add", csr_1, csr_2)
    np.testing.assert_allclose(
        adapt_backend_module(res_csr_add).require_symbol("to_dense")(),
        dense_1 + dense_2,
        rtol=1e-05,
        atol=1e-05,
    )
    res_csr_sub = execute_op(SparseGenerator, "Sub", csr_1, csr_2)
    np.testing.assert_allclose(
        adapt_backend_module(res_csr_sub).require_symbol("to_dense")(),
        dense_1 - dense_2,
        rtol=1e-05,
        atol=1e-05,
    )
    res_csc_add = execute_op(SparseGenerator, "add", csc_1, csc_2)
    np.testing.assert_allclose(
        adapt_backend_module(res_csc_add).require_symbol("to_dense")(),
        dense_1 + dense_2,
        rtol=1e-05,
        atol=1e-05,
    )
    res_csc_sub = execute_op(SparseGenerator, "sub", csc_1, csc_2)
    np.testing.assert_allclose(
        adapt_backend_module(res_csc_sub).require_symbol("to_dense")(),
        dense_1 - dense_2,
        rtol=1e-05,
        atol=1e-05,
    )
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
    (dense_adj, coo_adj) = _make_test_adj(4)
    attn_coo = kernels.sparse_softmax(coo_adj)
    assert isinstance(attn_coo, COOTensor)
    attn_dense = attn_coo.to_dense()
    for r in range(4):
        if np.any(dense_adj[r] != 0):
            assert np.isclose(np.sum(attn_dense[r]), 1.0, atol=1e-05)
        else:
            assert np.isclose(np.sum(attn_dense[r]), 0.0)
    empty_coo = COOTensor(indices=np.zeros((2, 0), dtype=np.int64), values=np.zeros(0, dtype=np.float32), shape=(4, 4))
    assert kernels.sparse_softmax(empty_coo).nnz == 0
    masked_softmax = kernels.sparse_softmax(coo_adj, mask=coo_adj)
    assert masked_softmax.nnz == coo_adj.nnz
    norm_adj = kernels.graph_norm_adjacency(coo_adj, add_self_loops=True)
    assert isinstance(norm_adj, COOTensor)
    assert norm_adj.shape == (4, 4)
    a_hat = dense_adj + np.eye(4, dtype=np.float32)
    deg_hat = np.sum(a_hat, axis=1)
    deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg_hat))
    expected_norm_adj = deg_inv_sqrt @ a_hat @ deg_inv_sqrt
    np.testing.assert_allclose(norm_adj.to_dense(), expected_norm_adj, rtol=1e-05, atol=1e-05)
    norm_adj_no_loops = kernels.graph_norm_adjacency(coo_adj, add_self_loops=False)
    assert norm_adj_no_loops.shape == (4, 4)
    node_features = np.array([[1.0, 0.5], [0.2, 0.8], [1.5, 0.1], [0.7, 0.9]], dtype=np.float32)
    agg_out = kernels.gnn_spmm_attention(coo_adj, node_features)
    assert extract_test_shape(agg_out) == (4, 2)
    np.testing.assert_allclose(ensure_ndarray(agg_out), dense_adj @ node_features, rtol=1e-05, atol=1e-05)
    agg_attn_out = kernels.gnn_spmm_attention(coo_adj, node_features, attention_weights=coo_adj)
    assert extract_test_shape(agg_attn_out) == (4, 2)
    dropped = kernels.sparse_dropout(coo_adj, drop_rate=0.5, training=True, seed=42)
    assert isinstance(dropped, COOTensor)
    assert dropped.shape == coo_adj.shape
    no_drop = kernels.sparse_dropout(coo_adj, drop_rate=0.5, training=False)
    assert no_drop.nnz == coo_adj.nnz
    zero_drop = kernels.sparse_dropout(coo_adj, drop_rate=0.0, training=True)
    assert zero_drop.nnz == coo_adj.nnz
    full_drop = kernels.sparse_dropout(coo_adj, drop_rate=1.0, training=True)
    assert full_drop.nnz == 0
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
    assert "from ml_switcheroo_compiler.backends.sparse import kernels as sp_kernels" in code
    gen_non_eff = SparseGenerator(g, memory_efficient=False)
    assert "# Memory-efficient sparse execution active" not in gen_non_eff.generate()
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
    g.nodes = NodeDict({"x": n_in, "out": n_sum})
    g.inputs = ["x"]
    g.outputs = ["out"]
    profiler = SparseProfiler()
    metrics = profiler.profile_graph(g, inputs={"x": [[1.0, 2.0], [3.0, 4.0]]}, num_iters=3, warmup_iters=1)
    assert "latencies" in metrics
    assert len(coerce_operand_list(metrics["latencies"])) == 3
    assert metrics["latency_ms"] >= 0.0
    assert metrics["p50_ms"] >= 0.0
    assert metrics["p95_ms"] >= 0.0
    assert metrics["p99_ms"] >= 0.0
    assert metrics["peak_memory_mb"] > 0.0
    with mock.patch("sys.platform", "linux"):
        assert _get_process_memory_mb() > 0.0
    empty_graph = IRGraph()
    metrics_empty = profiler.profile_graph(empty_graph, inputs={}, num_iters=2, warmup_iters=1)
    assert len(coerce_operand_list(metrics_empty["latencies"])) == 2

    class MockAddTensor(MockTestStub):
        """Test class MockAddTensor."""

        def __add__(self, other: BackendArg) -> BackendArg:
            """Test function __add__.

            Args:
                other (BackendArg): Test parameter `other`.

            Returns:
                BackendArg: Computed output value from `__add__`.
            """
            return self

    class MockNonAddTensor(MockTestStub):
        """Test class MockNonAddTensor."""

    metrics_add = profiler.profile_graph(empty_graph, inputs={"a": MockAddTensor(), "b": MockAddTensor()}, num_iters=2)
    assert len(coerce_operand_list(metrics_add["latencies"])) == 2
    metrics_non_add = profiler.profile_graph(
        empty_graph,
        inputs={"a": MockNonAddTensor(), "b": MockNonAddTensor()},
        num_iters=2,
    )
    assert len(coerce_operand_list(metrics_non_add["latencies"])) == 2


def test_sparse_eager_dispatch_coverage() -> None:
    """Verify eager dispatch error cases and unmapped operations."""
    (dense_adj, coo_adj) = _make_test_adj(4)
    soft_res = execute_op(SparseGenerator, "sparse_softmax", coo_adj)
    assert isinstance(soft_res, COOTensor)
    norm_res = execute_op(SparseGenerator, "graph_norm_adjacency", coo_adj)
    assert isinstance(norm_res, COOTensor)
    drop_res = execute_op(SparseGenerator, "sparse_dropout", coo_adj)
    assert isinstance(drop_res, COOTensor)
    (handled, _) = _dispatch_sparse_linalg("unknown_op", coo_adj, coo_adj)
    assert handled is False
    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(SparseGenerator, "NonexistentUnknownSparseOp", coo_adj)


class _TestSparseEdgeCasesAndBranchCoverageState(typing.TypedDict, total=False):
    """Typed shared state for `_test_sparse_edge_cases_and_branch_coverage` split helpers."""

    coo_adj: COOTensor
    csc_adj: CSCTensor
    csr_adj: CSRTensor
    dense_adj: np.ndarray[tuple[int, ...], np.dtype[np.float32]]
    empty_csc: CSCTensor
    empty_csr: CSRTensor
    fake_sparse_import: EagerBoundCallable
    g: LogicalGraph
    prepared: list[BackendArg]
    profiler: SparseProfiler
    sub_csr_res: CSRTensor


def _test_sparse_edge_cases_and_branch_coverage_part_1(_state: _TestSparseEdgeCasesAndBranchCoverageState) -> None:
    """Execute part 1 of test_sparse_edge_cases_and_branch_coverage.

    Args:
        _state (_TestSparseEdgeCasesAndBranchCoverageState): Shared test state dictionary.
    """
    (dense_adj, coo_adj) = _make_test_adj(4)
    csr_adj = coo_adj.to_csr()
    csc_adj = coo_adj.to_csc()
    (handled_dense_matmul, _) = _dispatch_sparse_linalg("MatMul", np.array([1]), np.array([2]))
    assert handled_dense_matmul is False
    (handled_add, _) = _dispatch_sparse_linalg("Add", np.array([1]), np.array([2]))
    assert handled_add is False
    (handled_mixed_add, _) = _dispatch_sparse_linalg("Add", csr_adj, csc_adj)
    assert handled_mixed_add is False
    (handled_coo_add, _) = _dispatch_sparse_linalg("Add", coo_adj, coo_adj)
    assert handled_coo_add is False
    (handled_sub, _) = _dispatch_sparse_linalg("Sub", np.array([1]), np.array([2]))
    assert handled_sub is False
    empty_coo = COOTensor(indices=np.zeros((2, 0), dtype=np.int64), values=np.zeros(0, dtype=np.float32), shape=(4, 4))
    empty_csr = empty_coo.to_csr()
    empty_csc = empty_coo.to_csc()
    with pytest.raises(ValueError, match="Shape mismatch"):
        kernels.csr_sub(csr_adj, CSRTensor.from_dense(np.ones((2, 2))))
    with pytest.raises(ValueError, match="Shape mismatch"):
        kernels.csc_sub(csc_adj, CSCTensor.from_dense(np.ones((2, 2))))
    sub_csr_res = kernels.csr_sub(empty_csr, csr_adj)
    assert isinstance(sub_csr_res, CSRTensor)
    _state.update({"coo_adj": coo_adj, "csc_adj": csc_adj, "csr_adj": csr_adj})
    _state.update({"dense_adj": dense_adj, "empty_csc": empty_csc, "empty_csr": empty_csr})
    _state.update({"sub_csr_res": sub_csr_res})


def _test_sparse_edge_cases_and_branch_coverage_part_2(_state: _TestSparseEdgeCasesAndBranchCoverageState) -> None:
    """Execute part 2 of test_sparse_edge_cases_and_branch_coverage.

    Args:
        _state (_TestSparseEdgeCasesAndBranchCoverageState): Shared test state dictionary.
    """
    (coo_adj, csc_adj, csr_adj, dense_adj) = (
        _state["coo_adj"],
        _state["csc_adj"],
        _state["csr_adj"],
        _state["dense_adj"],
    )
    (empty_csc, empty_csr, sub_csr_res) = (_state["empty_csc"], _state["empty_csr"], _state["sub_csr_res"])
    np.testing.assert_allclose(ensure_ndarray(sub_csr_res.to_dense()), -dense_adj)
    sub_csr_b_empty = kernels.csr_sub(csr_adj, empty_csr)
    assert sub_csr_b_empty is csr_adj
    sub_csc_res = kernels.csc_sub(empty_csc, csc_adj)
    assert isinstance(sub_csc_res, CSCTensor)
    np.testing.assert_allclose(ensure_ndarray(sub_csc_res.to_dense()), -dense_adj)
    sub_csc_b_empty = kernels.csc_sub(csc_adj, empty_csc)
    assert sub_csc_b_empty is csc_adj
    csr_100 = kernels.sparse_dropout(csr_adj, drop_rate=1.0)
    assert isinstance(csr_100, CSRTensor)
    assert csr_100.nnz == 0
    csc_100 = kernels.sparse_dropout(csc_adj, drop_rate=1.0)
    assert isinstance(csc_100, CSCTensor)
    assert csc_100.nnz == 0
    fake_rng = mock.MagicMock()
    fake_rng.uniform.return_value = np.ones(coo_adj.nnz)
    with mock.patch("numpy.random.default_rng", return_value=fake_rng):
        drop_all = kernels.sparse_dropout(coo_adj, drop_rate=0.5)
        assert drop_all.nnz == 0
    profiler = SparseProfiler()
    g = IRGraph()
    prepared = profiler._prepare_inputs(g, {"coo": coo_adj, "arr": np.array([1.0, 2.0]), "lst": [3.0, 4.0], "raw": 42})
    _state.update({"g": g, "prepared": prepared, "profiler": profiler})
    _state.update({"coo_adj": coo_adj, "csc_adj": csc_adj, "csr_adj": csr_adj})
    _state.update({"dense_adj": dense_adj, "empty_csc": empty_csc, "empty_csr": empty_csr})
    _state.update({"sub_csr_res": sub_csr_res})


def _test_sparse_edge_cases_and_branch_coverage_part_3(_state: _TestSparseEdgeCasesAndBranchCoverageState) -> None:
    """Execute part 3 of test_sparse_edge_cases_and_branch_coverage.

    Args:
        _state (_TestSparseEdgeCasesAndBranchCoverageState): Shared test state dictionary.
    """
    (g, prepared, profiler) = (_state["g"], _state["prepared"], _state["profiler"])
    assert len(prepared) == 4
    try:
        import sparse as pydata_sparse

        pydata_coo = pydata_sparse.COO.from_numpy(np.array([1.0, 2.0]))
    except ImportError:
        pydata_coo = mock.MagicMock(coords=np.array([[0, 1]]), data=np.array([1.0, 2.0]), shape=(2,))
    prepared_pydata = profiler._prepare_inputs(g, {"pydata": pydata_coo})
    assert len(prepared_pydata) == 1
    g_sum = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_out = IRNode(id="out", op_type="Sum", inputs=["x"], shape_metadata=[2, 2])
    g_sum.nodes = NodeDict({"x": n_in, "out": n_out})
    g_sum.inputs = ["x"]
    g_sum.outputs = ["out"]
    with mock.patch.object(profiler, "_compile_graph", side_effect=RuntimeError("compile failed")):
        metrics_fail = profiler.profile_graph(g_sum, inputs={"x": [[1.0, 2.0], [3.0, 4.0]]}, num_iters=2)
        assert len(metrics_fail["latencies"]) == 2
    real_import = builtins.__import__

    def fake_sparse_import(
        name: str,
        globals_map: Mapping[str, BackendArg] | None = None,
        locals_map: Mapping[str, BackendArg] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> types.ModuleType:
        """Test function fake_sparse_import.

        Args:
            name (str): Module name to import.
            globals_map (Mapping[str, BackendArg] | None): Caller globals mapping.
            locals_map (Mapping[str, BackendArg] | None): Caller locals mapping.
            fromlist (Sequence[str] | None): Requested attribute names.
            level (int): Relative import level.

        Returns:
            types.ModuleType: Computed output value from `fake_sparse_import`.

        Raises:
            ImportError: When raised during `fake_sparse_import` (no sparse module).
        """
        if name == "sparse":
            raise ImportError("no sparse module")
        return real_import(name, globals_map, locals_map, fromlist, level)

    _state.update({"fake_sparse_import": require_callable(fake_sparse_import)})
    _state.update({"g": g, "prepared": prepared, "profiler": profiler})


def _test_sparse_edge_cases_and_branch_coverage_part_4(_state: _TestSparseEdgeCasesAndBranchCoverageState) -> None:
    """Execute part 4 of test_sparse_edge_cases_and_branch_coverage.

    Args:
        _state (_TestSparseEdgeCasesAndBranchCoverageState): Shared test state dictionary.
    """
    fake_sparse_import = _state["fake_sparse_import"]
    import importlib

    with mock.patch("builtins.__import__", side_effect=fake_sparse_import):
        import ml_switcheroo_compiler.backends.sparse.profiler as sp_prof_mod

        importlib.reload(sp_prof_mod)
    import ml_switcheroo_compiler.backends.sparse.profiler as sp_prof_mod

    importlib.reload(sp_prof_mod)


def test_sparse_edge_cases_and_branch_coverage() -> None:
    """Verify edge conditions in eager linalg dispatch, kernels, and profiler."""
    _state: _TestSparseEdgeCasesAndBranchCoverageState = {}
    _test_sparse_edge_cases_and_branch_coverage_part_1(_state)
    _test_sparse_edge_cases_and_branch_coverage_part_2(_state)
    _test_sparse_edge_cases_and_branch_coverage_part_3(_state)
    _test_sparse_edge_cases_and_branch_coverage_part_4(_state)
    assert len(_state) > 0
