"""Exhaustive parity, control flow, vectorized ufunc, and profiling test suite for Numba backend."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import ml_switcheroo_compiler.backends.numba as nb_pkg
from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numba.eager import _unwrap_arg, execute_op
from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.backends.numba.profiler import NumbaProfiler, _get_process_memory_mb
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_numba_dynamic_control_flow_visitors_and_parity() -> None:
    """Verify Numba JIT emission and parity for Cond, WhileLoop, and Scan control flow."""
    g = IRGraph()
    gen = NumbaGenerator(g, fastmath=True, parallel=False)

    # 1. Cond visitor and eager execution
    cond_node = IRNode(id="cond_node", op_type="Cond")
    cond_code = gen.visit_Cond(cond_node, ["pred", "true_val", "false_val"])
    assert cond_code == "v_cond_node"
    assert "if pred:" in "\n".join(gen.code)
    assert "else:" in "\n".join(gen.code)

    # Eager Cond
    assert execute_op(NumbaGenerator, "Cond", True, 42, 99) == 42
    assert execute_op(NumbaGenerator, "Cond", False, 42, 99) == 99
    assert execute_op(NumbaGenerator, "Cond", True, lambda: 100, lambda: 200) == 100
    assert execute_op(NumbaGenerator, "Cond", False, lambda: 100, lambda: 200) == 200

    # 2. WhileLoop visitor and eager execution
    gen_while = NumbaGenerator(g, fastmath=True)
    while_node = IRNode(id="while_node", op_type="WhileLoop", attributes={"max_iters": 5})
    while_code = gen_while.visit_WhileLoop(while_node, ["init_val"])
    assert while_code == "v_while_node"
    assert "while iter_while_node < 5:" in "\n".join(gen_while.code)

    # Eager WhileLoop
    assert execute_op(NumbaGenerator, "WhileLoop", 10, max_iters=5) == 15

    # 3. Scan visitor and eager execution
    gen_scan = NumbaGenerator(g, fastmath=True)
    scan_node = IRNode(id="scan_node", op_type="Scan")
    scan_code = gen_scan.visit_Scan(scan_node, ["0", "xs"])
    assert scan_code == "v_scan_node"
    assert "for i_scan_node in range(len(xs)):" in "\n".join(gen_scan.code)

    # Eager Scan
    scan_res = execute_op(NumbaGenerator, "Scan", 0, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(scan_res, [1.0, 3.0, 6.0, 10.0])


def test_numba_vectorized_ufunc_generation_and_compilation() -> None:
    """Verify Numba @vectorize ufunc code synthesis, compilation, and execution."""
    g = IRGraph()
    gen = NumbaGenerator(g, fastmath=True)

    # 1. Binary vectorized ufunc: add_squared (x + y**2)
    source = gen.generate_vectorized_ufunc(
        op_name="add_squared",
        scalar_expr="x + y * y",
        arg_names=["x", "y"],
        target="cpu",
    )
    assert "@nb.vectorize" in source
    assert "def add_squared(x, y):" in source

    # Compile binary ufunc
    ufunc = gen.compile_vectorized_ufunc(
        op_name="add_squared",
        scalar_expr="x + y * y",
        arg_names=["x", "y"],
    )
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    expected = a + b * b
    np.testing.assert_allclose(ufunc(a, b), expected)

    # 2. Unary vectorized ufunc: relu_scale (max(0, x) * 2)
    ufunc_unary = gen.compile_vectorized_ufunc(
        op_name="relu_scale",
        scalar_expr="x * 2.0 if x > 0.0 else 0.0",
        arg_names=["x"],
    )
    x = np.array([-1.0, 0.0, 2.5], dtype=np.float32)
    expected_unary = np.array([0.0, 0.0, 5.0], dtype=np.float32)
    np.testing.assert_allclose(ufunc_unary(x), expected_unary)

    # 3. Failing compilation branch
    with mock.patch("builtins.exec"):
        with pytest.raises(RuntimeError, match="failed to produce callable"):
            gen.compile_vectorized_ufunc("broken_fn", "x")


def test_numba_eager_dispatch_and_tensor_unwrap() -> None:
    """Verify Tensor argument unwrapping, registry fallback, and direct ops in eager dispatch."""
    cfg = TensorConfig((3,), "float32", "cpu")
    t1 = Tensor(np.array([1.0, 2.0, 3.0], dtype=np.float32), cfg)
    t2 = Tensor(np.array([4.0, 5.0, 6.0], dtype=np.float32), cfg)

    # Tensor unwrapping
    assert _unwrap_arg(t1) is not None
    assert _unwrap_arg(5) == 5

    res_add = execute_op(NumbaGenerator, "Add", t1, t2)
    np.testing.assert_allclose(res_add, [5.0, 7.0, 9.0])

    # Direct numpy function
    res_sin = execute_op(NumbaGenerator, "sin", np.array([0.0, np.pi / 2.0]))
    np.testing.assert_allclose(res_sin, [0.0, 1.0], atol=1e-5)

    # Eager registry fallback
    def mock_nb_custom(mod: object, *args: object, **kwargs: object) -> str:
        del mod, args, kwargs
        return "custom_numba_res"

    with mock.patch.dict(global_eager_registry._registry, {"CustomNumbaOp": mock_nb_custom}):
        assert execute_op(NumbaGenerator, "CustomNumbaOp", 1) == "custom_numba_res"

    # Unsupported operation and non-callable attribute
    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(NumbaGenerator, "CompletelyUnknownNumbaOpXYZ", t1)

    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(NumbaGenerator, "pi", t1)


def test_numba_profiler_and_memory_metrics() -> None:
    """Verify NumbaProfiler graph profiling, latency statistics, and memory tracking."""
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_sum = IRNode(id="out", op_type="Sum", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_in, "out": n_sum}
    g.inputs = ["x"]
    g.outputs = ["out"]

    profiler = NumbaProfiler()
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

    # Profile with fallback summation
    metrics_sum = profiler.profile_graph(
        empty_graph,
        inputs={"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])},
        num_iters=2,
    )
    assert len(metrics_sum["latencies"]) == 2

    # Failing compile_graph fallback
    with mock.patch.object(profiler, "_compile_graph", side_effect=RuntimeError("compile failed")):
        metrics_fallback = profiler.profile_graph(g, inputs={"x": [[1.0, 2.0], [3.0, 4.0]]}, num_iters=2)
        assert len(metrics_fallback["latencies"]) == 2

    # Profiler import fallback when numba is not installed
    real_import = __import__

    def fake_nb_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "numba":
            raise ImportError("no numba")
        return real_import(name, *args, **kwargs)

    import importlib

    with mock.patch("builtins.__import__", side_effect=fake_nb_import):
        import ml_switcheroo_compiler.backends.numba.profiler as nb_prof_mod

        importlib.reload(nb_prof_mod)

    # Restore profiler
    import ml_switcheroo_compiler.backends.numba.profiler as nb_prof_mod

    importlib.reload(nb_prof_mod)


def test_numba_generator_aot_and_classmethods() -> None:
    """Verify NumbaGenerator AOT execution and attached classmethods."""
    g = IRGraph()
    n_in = IRNode(id="x", op_type="Input", inputs=[], shape_metadata=[2, 2])
    n_abs = IRNode(id="out", op_type="Abs", inputs=["x"], shape_metadata=[2, 2])
    g.nodes = {"x": n_in, "out": n_abs}
    g.inputs = ["x"]
    g.outputs = ["out"]

    gen = NumbaGenerator(g)
    artifact = gen._compile_aot_impl(g)
    assert callable(artifact)

    input_arr = np.array([[-1.0, 2.0], [-3.0, 4.0]], dtype=np.float32)
    res = artifact(input_arr)
    np.testing.assert_allclose(res, np.abs(input_arr))

    # Classmethods on NumbaGenerator and package exports
    assert nb_pkg.zeros((2, 2)).shape == (2, 2)
    assert NumbaGenerator.zeros((2, 2)).shape == (2, 2)
    assert nb_pkg.array([1.0, 2.0]).shape == (2,)
    assert NumbaGenerator.array([1.0, 2.0]).shape == (2,)
    assert nb_pkg.asarray([1.0, 2.0]).shape == (2,)
    assert NumbaGenerator.asarray([1.0, 2.0]).shape == (2,)
    assert nb_pkg.item(np.array([42.0])) == 42.0
    assert NumbaGenerator.item(np.array([42.0])) == 42.0
    assert nb_pkg.execute_op(NumbaGenerator, "Add", 1, 2) == 3
