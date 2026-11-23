"""Unit tests for Numba parallel loop emission and code generation."""

from __future__ import annotations

import typing

import numpy as np
import pytest
from ml_switcheroo_ir import IRGraph, IRNode, LogicalGraph

from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.backends.numpy.eager import ensure_ndarray
from ml_switcheroo_compiler.core.shape import OpValue
from tests.testing_utils import EagerTensorProtocol, GenericNDArray, extract_test_shape, require_callable, unpack_pair

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


@pytest.fixture(autouse=True)
def _ensure_numba_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a NumPy-backed numba stub when numba is not installed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture for temporary attribute and environment overrides.
    """
    import importlib.util
    import sys
    from unittest.mock import MagicMock

    if importlib.util.find_spec("numba") is None:
        mock_nb = MagicMock()
        mock_nb.njit.side_effect = lambda *a, **kw: a[0] if a and callable(a[0]) else (lambda fn: fn)
        mock_nb.prange = range
        mock_nb.np.numpy_support.np = np
        mock_nb.vectorize.side_effect = lambda *a, **kw: lambda fn: np.vectorize(fn)
        monkeypatch.setitem(sys.modules, "numba", mock_nb)


class _TestNumbaParallelLoopEmissionState(typing.TypedDict, total=False):
    """Typed shared state for `_test_numba_parallel_loop_emission` split helpers."""

    fn_conv: EagerTensorProtocol | OpValue
    g_bmm: LogicalGraph
    g_matmul: LogicalGraph
    gen_nopar: NumbaGenerator
    w_conv: GenericNDArray
    x_conv: GenericNDArray


def _test_numba_parallel_loop_emission_part_1(_state: _TestNumbaParallelLoopEmissionState) -> None:
    """Execute part 1 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    g_matmul = IRGraph()
    g_matmul.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[])
    g_matmul.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[])
    g_matmul.nodes["mm"] = IRNode(id="mm", op_type="MatMul", inputs=["a", "b"])
    g_matmul.inputs = ["a", "b"]
    g_matmul.outputs = ["mm"]

    gen_mm = NumbaGenerator(g_matmul, fastmath=True, parallel=True)
    code_mm = gen_mm.generate()
    assert "nb.njit(fastmath=True, parallel=True)" in code_mm
    assert "for _i in nb.prange(" in code_mm

    gen_mm_strict = NumbaGenerator(g_matmul, fastmath=True, parallel=True, emit_safe_fallback=False)
    assert "@nb.njit(fastmath=True, parallel=True)" in gen_mm_strict.generate()

    fn_mm = gen_mm.compile_fn()
    a_mat = np.ones((4, 4), dtype=np.float32)
    b_mat = np.ones((4, 4), dtype=np.float32)
    res_mm = fn_mm([a_mat, b_mat])
    np.testing.assert_allclose(ensure_ndarray(res_mm), a_mat @ b_mat)

    # BatchMatMul
    g_bmm = IRGraph()
    g_bmm.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[])
    g_bmm.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[])
    g_bmm.nodes["bmm"] = IRNode(id="bmm", op_type="BatchMatMul", inputs=["a", "b"])
    g_bmm.inputs = ["a", "b"]
    _state.update({"g_bmm": g_bmm, "g_matmul": g_matmul})


def _test_numba_parallel_loop_emission_part_2(_state: _TestNumbaParallelLoopEmissionState) -> None:
    """Execute part 2 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    g_bmm = _state["g_bmm"]
    pytest.MonkeyPatch().setattr(g_bmm, "outputs", ["bmm"], raising=False)

    gen_bmm = NumbaGenerator(g_bmm, fastmath=True, parallel=True)
    code_bmm = gen_bmm.generate()
    assert "for _b in nb.prange(" in code_bmm

    fn_bmm = gen_bmm.compile_fn()
    a_batch = np.ones((2, 3, 4), dtype=np.float32)
    b_batch = np.ones((2, 4, 5), dtype=np.float32)
    res_bmm = fn_bmm([a_batch, b_batch])
    np.testing.assert_allclose(ensure_ndarray(res_bmm), a_batch @ b_batch)

    # Conv2D
    g_conv = IRGraph()
    g_conv.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[])
    g_conv.nodes["w"] = IRNode(id="w", op_type="Input", inputs=[])
    g_conv.nodes["conv"] = IRNode(id="conv", op_type="Conv2D", inputs=["x", "w"])
    g_conv.inputs = ["x", "w"]
    g_conv.outputs = ["conv"]

    gen_conv = NumbaGenerator(g_conv, fastmath=True, parallel=True)
    code_conv = gen_conv.generate()
    assert "for _b in nb.prange(" in code_conv
    assert "for _h in nb.prange(_h_out):" in code_conv

    fn_conv = gen_conv.compile_fn()
    x_conv = np.ones((2, 1, 4, 4), dtype=np.float32)
    w_conv = np.ones((1, 1, 3, 3), dtype=np.float32)
    _state.update({"fn_conv": fn_conv})
    _state.update({"w_conv": w_conv, "x_conv": x_conv})
    _state.update({"g_bmm": g_bmm})


def _test_numba_parallel_loop_emission_part_3(_state: _TestNumbaParallelLoopEmissionState) -> None:
    """Execute part 3 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    fn_conv, g_matmul, w_conv, x_conv = _state["fn_conv"], _state["g_matmul"], _state["w_conv"], _state["x_conv"]
    res_conv = require_callable(fn_conv)([x_conv, w_conv])
    assert res_conv.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(res_conv, 9.0)

    # MaxPool2D and AvgPool2D
    g_pool = IRGraph()
    g_pool.nodes["x"] = IRNode(id="x", op_type="Input", inputs=[])
    g_pool.nodes["maxp"] = IRNode(id="maxp", op_type="MaxPool2D", inputs=["x"])
    g_pool.nodes["avgp"] = IRNode(id="avgp", op_type="AvgPool2D", inputs=["x"])
    g_pool.inputs = ["x"]
    g_pool.outputs = ["maxp", "avgp"]

    gen_pool = NumbaGenerator(g_pool, fastmath=True, parallel=True)
    code_pool = gen_pool.generate()
    assert "for _b in nb.prange(" in code_pool
    assert "for _h in nb.prange(_h_out):" in code_pool

    fn_pool = gen_pool.compile_fn()
    x_pool = np.arange(32, dtype=np.float32).reshape(2, 1, 4, 4)
    res_maxp, res_avgp = unpack_pair(fn_pool([x_pool]))
    assert extract_test_shape(res_maxp) == (2, 1, 2, 2)
    assert extract_test_shape(res_avgp) == (2, 1, 2, 2)

    # Non-parallel fallback branch coverage
    gen_nopar = NumbaGenerator(g_matmul, fastmath=False, parallel=False)
    code_nopar = gen_nopar.generate()
    assert "np.matmul" in code_nopar

    gen_nopar.visit_Prod(IRNode(id="p", op_type="Prod", inputs=["a"]), ["a"])
    _state.update({"gen_nopar": gen_nopar})
    _state.update({"fn_conv": fn_conv, "g_matmul": g_matmul, "w_conv": w_conv})
    _state.update({"x_conv": x_conv})


def _test_numba_parallel_loop_emission_part_4(_state: _TestNumbaParallelLoopEmissionState) -> None:
    """Execute part 4 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    gen_nopar = _state["gen_nopar"]
    assert any("np.prod" in line for line in gen_nopar.code)
    gen_nopar.visit_Mean(IRNode(id="m", op_type="Mean", inputs=["a"]), ["a"])
    assert any("np.mean" in line for line in gen_nopar.code)
    gen_nopar.visit_Max(IRNode(id="mx", op_type="Max", inputs=["a"]), ["a"])
    assert any("np.max" in line for line in gen_nopar.code)
    gen_nopar.visit_Min(IRNode(id="mn", op_type="Min", inputs=["a"]), ["a"])
    assert any("np.min" in line for line in gen_nopar.code)
    gen_nopar.visit_BatchMatMul(IRNode(id="bmm", op_type="BatchMatMul", inputs=["a", "b"]), ["a", "b"])
    assert any("np.matmul" in line for line in gen_nopar.code)
    gen_nopar.visit_Conv2D(IRNode(id="cv", op_type="Conv2D", inputs=["x", "w"]), ["x", "w"])
    _state.update({"gen_nopar": gen_nopar})


def _test_numba_parallel_loop_emission_part_5(_state: _TestNumbaParallelLoopEmissionState) -> None:
    """Execute part 5 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    gen_nopar = _state["gen_nopar"]
    assert any("np.zeros" in line for line in gen_nopar.code)
    gen_nopar.visit_MaxPool2D(IRNode(id="mp", op_type="MaxPool2D", inputs=["x"]), ["x"])
    assert any("np.zeros" in line for line in gen_nopar.code)
    gen_nopar.visit_AvgPool2D(IRNode(id="ap", op_type="AvgPool2D", inputs=["x"]), ["x"])
    assert any("np.zeros" in line for line in gen_nopar.code)


def test_numba_parallel_loop_emission() -> None:
    """Verify Numba automatic parallel loop emission with @nb.njit(fastmath=True, parallel=True)."""
    _state: _TestNumbaParallelLoopEmissionState = {}
    _test_numba_parallel_loop_emission_part_1(_state)
    _test_numba_parallel_loop_emission_part_2(_state)
    _test_numba_parallel_loop_emission_part_3(_state)
    _test_numba_parallel_loop_emission_part_4(_state)
    _test_numba_parallel_loop_emission_part_5(_state)
    assert len(_state) > 0
