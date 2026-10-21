"""Unit tests for Numba parallel loop emission and code generation."""

from __future__ import annotations

import typing

import numpy as np

from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode
from tests.testing_utils import as_callable, as_tensor

if typing.TYPE_CHECKING:
    from ml_switcheroo_compiler.backends.eager_registry import EagerTensorProtocol


def _test_numba_parallel_loop_emission_part_1(_state: dict[str, EagerTensorProtocol]) -> None:
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
    as_callable(np.testing.assert_allclose)(res_mm, a_mat @ b_mat)

    # BatchMatMul
    g_bmm = IRGraph()
    g_bmm.nodes["a"] = IRNode(id="a", op_type="Input", inputs=[])
    g_bmm.nodes["b"] = IRNode(id="b", op_type="Input", inputs=[])
    g_bmm.nodes["bmm"] = IRNode(id="bmm", op_type="BatchMatMul", inputs=["a", "b"])
    g_bmm.inputs = ["a", "b"]
    getattr(_state, "update")({"a_mat": a_mat, "b_mat": b_mat, "code_mm": code_mm, "fn_mm": fn_mm})
    getattr(_state, "update")({"g_bmm": g_bmm, "g_matmul": g_matmul, "gen_mm": gen_mm, "gen_mm_strict": gen_mm_strict})
    getattr(_state, "update")({"res_mm": res_mm})
    _state.update(locals())


def _test_numba_parallel_loop_emission_part_2(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 2 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    g_bmm = _state["g_bmm"]
    setattr(g_bmm, "outputs", ["bmm"])

    gen_bmm = as_callable(NumbaGenerator)(g_bmm, fastmath=True, parallel=True)
    code_bmm = gen_bmm.generate()
    assert "for _b in nb.prange(" in code_bmm

    fn_bmm = gen_bmm.compile_fn()
    a_batch = np.ones((2, 3, 4), dtype=np.float32)
    b_batch = np.ones((2, 4, 5), dtype=np.float32)
    res_bmm = fn_bmm([a_batch, b_batch])
    as_callable(np.testing.assert_allclose)(res_bmm, a_batch @ b_batch)

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
    getattr(_state, "update")({"a_batch": a_batch, "b_batch": b_batch, "code_bmm": code_bmm, "code_conv": code_conv})
    getattr(_state, "update")({"fn_bmm": fn_bmm, "fn_conv": fn_conv, "g_conv": g_conv, "gen_bmm": gen_bmm})
    getattr(_state, "update")({"gen_conv": gen_conv, "res_bmm": res_bmm, "w_conv": w_conv, "x_conv": x_conv})
    _state.update(locals())


def _test_numba_parallel_loop_emission_part_3(_state: dict[str, EagerTensorProtocol]) -> None:
    """Execute part 3 of test_numba_parallel_loop_emission.

    Args:
        _state: Shared test state dictionary.
    """
    fn_conv, g_matmul, w_conv, x_conv = _state["fn_conv"], _state["g_matmul"], _state["w_conv"], _state["x_conv"]
    res_conv = fn_conv([x_conv, w_conv])
    assert res_conv.shape == (2, 1, 2, 2)
    as_callable(np.testing.assert_allclose)(res_conv, 9.0)

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
    res_maxp, res_avgp = fn_pool([x_pool])
    assert as_tensor(getattr(res_maxp, "shape")) == (2, 1, 2, 2)
    assert as_tensor(getattr(res_avgp, "shape")) == (2, 1, 2, 2)

    # Non-parallel fallback branch coverage
    gen_nopar = as_callable(NumbaGenerator)(g_matmul, fastmath=False, parallel=False)
    code_nopar = gen_nopar.generate()
    assert "np.matmul" in code_nopar

    gen_nopar.visit_Prod(IRNode(id="p", op_type="Prod", inputs=["a"]), ["a"])
    getattr(_state, "update")({"code_nopar": code_nopar, "code_pool": code_pool, "fn_pool": fn_pool, "g_pool": g_pool})
    getattr(_state, "update")({"gen_nopar": gen_nopar, "gen_pool": gen_pool, "res_avgp": res_avgp, "res_conv": res_conv})
    getattr(_state, "update")({"res_maxp": res_maxp, "x_pool": x_pool})
    _state.update(locals())


def _test_numba_parallel_loop_emission_part_4(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state.update(locals())


def _test_numba_parallel_loop_emission_part_5(_state: dict[str, EagerTensorProtocol]) -> None:
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
    _state: dict[str, EagerTensorProtocol] = {}
    _test_numba_parallel_loop_emission_part_1(_state)
    _test_numba_parallel_loop_emission_part_2(_state)
    _test_numba_parallel_loop_emission_part_3(_state)
    _test_numba_parallel_loop_emission_part_4(_state)
    _test_numba_parallel_loop_emission_part_5(_state)
    assert len(_state) > 0
