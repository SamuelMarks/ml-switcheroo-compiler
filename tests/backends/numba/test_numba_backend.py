"""Tests for Numba backend generator, eager execution, and type conversions."""

from unittest.mock import patch

import numpy as np
import pytest

from ml_switcheroo_compiler.backends.eager_registry import global_eager_registry
from ml_switcheroo_compiler.backends.numba.eager import execute_op
from ml_switcheroo_compiler.backends.numba.generator import NumbaGenerator
from ml_switcheroo_compiler.backends.numba.types import array, asarray, item, zeros
from ml_switcheroo_compiler.backends.registry import BackendRegistry
from ml_switcheroo_compiler.core.errors import BackendNotSupportedError
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


class DummyGraph:
    """Dummy IRGraph for generator testing."""

    def __init__(self) -> None:
        """Initialize empty dummy graph."""
        self.nodes: list[IRNode] = []
        self.inputs: list[str] = ["x"]
        self.outputs: list[str] = ["out"]


def test_numba_backend_registered() -> None:
    """Test that numba backend is properly registered in BackendRegistry."""
    gen_cls = BackendRegistry.get("numba")
    assert gen_cls is NumbaGenerator


def test_numba_generator_initialization_and_options() -> None:
    """Test NumbaGenerator with various JIT flag combinations."""
    g = DummyGraph()
    gen = NumbaGenerator(g, fastmath=True, parallel=True, nogil=True, emit_safe_fallback=True)
    assert gen.get_fallback_prefix() == "np"
    assert gen.get_helper_functions() == []

    opts_str = gen._format_jit_options()
    assert "fastmath=True" in opts_str
    assert "parallel=True" in opts_str
    assert "nogil=True" in opts_str

    gen_none = NumbaGenerator(g, fastmath=False, parallel=False, nogil=False)
    assert gen_none._format_jit_options() == "()"


def test_numba_generator_generate() -> None:
    """Test source code generation for Numba functions."""
    g = DummyGraph()
    gen = NumbaGenerator(g, fastmath=True, parallel=False, nogil=False, emit_safe_fallback=True)
    code = gen.generate()
    assert "import numba as nb" in code
    assert "_njit = nb.njit(fastmath=True) if nb is not None else (lambda fn: fn)" in code
    assert "@_njit" in code
    assert "def evaluate(args):" in code

    gen_strict = NumbaGenerator(g, fastmath=False, parallel=True, emit_safe_fallback=False)
    code_strict = gen_strict.generate()
    assert "@nb.njit(parallel=True)" in code_strict


def test_numba_generator_compile_fn() -> None:
    """Test compiling generated function to executable callable."""
    graph = IRGraph()
    graph.nodes["in1"] = IRNode(id="in1", op_type="Input", inputs=[])
    graph.nodes["in2"] = IRNode(id="in2", op_type="Input", inputs=[])
    node = IRNode(id="out", op_type="Add", inputs=["in1", "in2"], attributes={})
    graph.nodes["out"] = node
    graph.inputs = ["in1", "in2"]
    graph.outputs = ["out"]

    gen = NumbaGenerator(graph, fastmath=True, emit_safe_fallback=True)
    fn = gen.compile_fn()
    assert callable(fn)

    res = fn([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
    np.testing.assert_allclose(res, np.array([4.0, 6.0]))


def test_numba_generator_compile_fn_error() -> None:
    """Test compile_fn when generated code does not produce the expected callable."""
    g = DummyGraph()
    gen = NumbaGenerator(g)
    with patch.object(gen, "generate", return_value="x = 42"):
        with pytest.raises(RuntimeError, match="Generated code did not produce callable"):
            gen.compile_fn()


def test_numba_generator_generic_visit() -> None:
    """Test generic_visit method of NumbaGenerator."""
    g = DummyGraph()
    gen = NumbaGenerator(g)
    node = IRNode(id="n1", op_type="Sin", inputs=["x"], attributes={})
    res = gen.generic_visit(node, ["x"])
    assert res == "np.sin(x)"


def test_numba_types() -> None:
    """Test Numba type conversion and tensor creation helpers."""
    z = zeros(type, (2, 3))
    assert isinstance(z, np.ndarray)
    assert z.shape == (2, 3)

    arr = array(type, [1.0, 2.0])
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr, [1.0, 2.0])

    arr_dtype = array(type, [1.0, 2.0], dtype="float64")
    assert arr_dtype.dtype == np.float64

    as_arr = asarray(type, [3.0, 4.0])
    assert isinstance(as_arr, np.ndarray)

    val = item(type, np.array([5.5]))
    assert val == 5.5


def test_numba_eager_execute_op() -> None:
    """Test eager operation dispatch for Numba backend."""
    res_abs = execute_op(type, "Abs", [-3.0, 4.0])
    np.testing.assert_allclose(res_abs, [3.0, 4.0])

    # Test non-list argument (direct ndarray)
    res_add = execute_op(type, "Add", np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    np.testing.assert_allclose(res_add, [4.0, 6.0])

    # Test object where asarray raises exception inside schema operation
    with patch("ml_switcheroo_compiler.backends.numba.eager.np.asarray", side_effect=ValueError("bad array")):
        res_fallback_arg = execute_op(type, "Abs", [1, 2])
        np.testing.assert_allclose(res_fallback_arg, [1, 2])

    # Test dispatch_eager_op raising BackendNotSupportedError falling back to global_eager_registry
    def fallback_op(backend_mod: object, *args: object, **kwargs: object) -> str:
        return "fallback_op_result"

    global_eager_registry.register("Abs")(fallback_op)
    with patch("ml_switcheroo_compiler.backends.mapping_loader.dispatch_eager_op", side_effect=BackendNotSupportedError("mock error")):
        res_fallback = execute_op(type, "Abs", [-1.0])
        assert res_fallback == "fallback_op_result"
    del global_eager_registry._registry["Abs"]

    def custom_op(backend_mod: object, *args: object, **kwargs: object) -> str:
        return "custom_numba_eager_ok"

    global_eager_registry.register("TestNumbaCustomOp")(custom_op)
    res_custom = execute_op(type, "TestNumbaCustomOp")
    assert res_custom == "custom_numba_eager_ok"

    with pytest.raises(BackendNotSupportedError, match="not implemented"):
        execute_op(type, "CompletelyUnknownNumbaOp12345")


def test_numba_init_import_guard() -> None:
    """Test package import guard when numba is not installed."""
    import importlib
    import sys

    import ml_switcheroo_compiler.backends.numba as nb_pkg

    assert hasattr(nb_pkg, "NumbaGenerator")
    assert hasattr(nb_pkg, "execute_op")
    assert hasattr(nb_pkg, "array")

    # Test find_spec raising ValueError
    with patch("importlib.util.find_spec", side_effect=ValueError):
        importlib.reload(nb_pkg)

    # Test import guard raising ImportError outside sphinx/pytest
    with patch("importlib.util.find_spec", return_value=None):
        with patch.dict(sys.modules):
            if "pytest" in sys.modules:
                del sys.modules["pytest"]
            if "sphinx" in sys.modules:
                del sys.modules["sphinx"]
            with pytest.raises(ImportError, match="requires the 'numba' library"):
                importlib.reload(nb_pkg)
    # Reload after exiting patch to restore clean state
    importlib.reload(nb_pkg)


def test_numba_control_flow_while_and_fori_loops() -> None:
    """Verify Numba visitors for control flow loops (WhileLoop, ForiLoop)."""
    g = DummyGraph()

    # Standard loop generation
    gen = NumbaGenerator(g, parallel=False)
    node_while = IRNode(id="while_1", op_type="WhileLoop", inputs=["init_x"], attributes={"max_iters": 10})
    out_while = gen.visit_WhileLoop(node_while, ["init_x"])
    assert out_while == "v_while_1"
    assert any("while iter_while_1 < 10:" in line for line in gen.code)

    node_fori = IRNode(id="fori_1", op_type="ForiLoop", inputs=["init_y"], attributes={"lower": 0, "upper": 20, "step": 2})
    out_fori = gen.visit_ForiLoop(node_fori, ["init_y"])
    assert out_fori == "v_fori_1"
    assert any("for idx_fori_1 in range(0, 20, 2):" in line for line in gen.code)

    # Parallel loop generation with nb.prange
    gen_par = NumbaGenerator(g, parallel=True)
    out_fori_par = gen_par.visit_ForiLoop(node_fori, ["init_y"])
    assert any("for idx_fori_1 in nb.prange(0, 20, 2):" in line for line in gen_par.code)


def test_numba_parallel_reduction_lowering_prange() -> None:
    """Verify fastmath and parallel loop lowering (nb.prange) for reduction operations."""
    g = DummyGraph()

    # Parallel mode with nb.prange
    gen_par = NumbaGenerator(g, fastmath=True, parallel=True)

    n_sum = IRNode(id="sum_1", op_type="Sum", inputs=["arr_in"])
    out_sum = gen_par.visit_Sum(n_sum, ["arr_in"])
    assert out_sum == "v_sum_1"
    assert any("for _i in nb.prange(arr_in.size):" in line for line in gen_par.code)

    n_prod = IRNode(id="prod_1", op_type="Prod", inputs=["arr_in"])
    out_prod = gen_par.visit_Prod(n_prod, ["arr_in"])
    assert out_prod == "v_prod_1"

    n_mean = IRNode(id="mean_1", op_type="Mean", inputs=["arr_in"])
    out_mean = gen_par.visit_Mean(n_mean, ["arr_in"])
    assert out_mean == "v_mean_1"

    n_max = IRNode(id="max_1", op_type="Max", inputs=["arr_in"])
    out_max = gen_par.visit_Max(n_max, ["arr_in"])
    assert out_max == "v_max_1"

    n_min = IRNode(id="min_1", op_type="Min", inputs=["arr_in"])
    out_min = gen_par.visit_Min(n_min, ["arr_in"])
    assert out_min == "v_min_1"

    # Non-parallel fallback
    gen_seq = NumbaGenerator(g, parallel=False)
    out_sum_seq = gen_seq.visit_Sum(n_sum, ["arr_in"])
    assert any("v_sum_1 = np.sum(arr_in)" in line for line in gen_seq.code)


def test_numba_expanded_eager_operations_parity() -> None:
    """Verify expanded Numba eager operations match NumPy eager results."""
    arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    arr2 = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

    # Trig and Hyperbolic
    np.testing.assert_allclose(execute_op(type, "Cos", arr1), np.cos(arr1))
    np.testing.assert_allclose(execute_op(type, "Tan", arr1), np.tan(arr1))
    np.testing.assert_allclose(execute_op(type, "Tanh", arr1), np.tanh(arr1))

    # Comparisons and Logic
    np.testing.assert_array_equal(execute_op(type, "Greater", arr1, arr2), np.greater(arr1, arr2))
    np.testing.assert_array_equal(execute_op(type, "Less", arr1, arr2), np.less(arr1, arr2))
    np.testing.assert_array_equal(execute_op(type, "Equal", arr1, arr1), np.equal(arr1, arr1))

    # Reductions
    np.testing.assert_allclose(execute_op(type, "Var", arr1), np.var(arr1))
    np.testing.assert_allclose(execute_op(type, "Std", arr1), np.std(arr1))
    np.testing.assert_allclose(execute_op(type, "Cumsum", arr1), np.cumsum(arr1))

    # Array creation and shape
    np.testing.assert_allclose(execute_op(type, "Zeros", (2, 2)), np.zeros((2, 2)))
    np.testing.assert_allclose(execute_op(type, "Ones", (3,)), np.ones(3))
    np.testing.assert_allclose(execute_op(type, "Arange", 5), np.arange(5))


def test_numba_aot_compilation_and_runner_execution() -> None:
    """Verify Numba ahead-of-time compilation, successful dispatch, and fallback branches."""
    # 1. Normal compilation and successful execution with single output
    g = IRGraph(name="test_nb_aot")
    g.nodes = {
        "x": IRNode(id="x", op_type="Input", shape_metadata=(4,)),
        "y": IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(4,)),
    }
    g.inputs = ["x"]
    g.outputs = ["y"]
    gen = NumbaGenerator(g)
    artifact = gen.compile_aot(g)
    assert artifact.metadata is not None
    assert artifact.metadata["backend"] == "numba"
    assert artifact["backend"] == "numba"

    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    res = artifact(arr)
    np.testing.assert_allclose(res, arr + arr)

    # 2. Multi-output execution with extra unused arguments
    g_multi = IRGraph(name="test_nb_aot_multi")
    g_multi.nodes = {
        "x": IRNode(id="x", op_type="Input", shape_metadata=(4,)),
        "y": IRNode(id="y", op_type="Add", inputs=["x", "x"], shape_metadata=(4,)),
        "z": IRNode(id="z", op_type="Mul", inputs=["x", "y"], shape_metadata=(4,)),
    }
    g_multi.inputs = ["x"]
    g_multi.outputs = ["y", "z"]
    gen_multi = NumbaGenerator(g_multi)
    artifact_multi = gen_multi.compile_aot(g_multi)
    res_multi = artifact_multi(arr, "extra_arg")
    assert isinstance(res_multi, tuple) and len(res_multi) == 2
    np.testing.assert_allclose(res_multi[0], arr + arr)
    np.testing.assert_allclose(res_multi[1], arr * (arr + arr))

    # 3. Fallback execution when compiled_fn fails to exec (syntax error in source_code)
    with patch.object(gen, "generate", return_value="def invalid python syntax !!!"):
        artifact_bad = gen._compile_aot_impl(g)
        res_fallback = artifact_bad(arr, "extra_arg")
        np.testing.assert_allclose(res_fallback, arr + arr)

    # 4. Fallback execution with empty outputs
    g_empty = IRGraph(name="test_nb_aot_empty")
    g_empty.nodes = {"x": IRNode(id="x", op_type="Input", shape_metadata=(4,))}
    g_empty.inputs = ["x"]
    g_empty.outputs = []
    with patch.object(gen, "generate", return_value="def invalid python syntax !!!"):
        artifact_empty = gen._compile_aot_impl(g_empty)
        res_empty = artifact_empty(arr)
        assert isinstance(res_empty, dict)

    # 5. Fallback execution when compiled_fn raises exception inside runner
    def exploding_fn(*args: object) -> None:
        raise RuntimeError("simulated numba execution failure")

    with patch.object(gen, "generate", return_value="def evaluate(args): pass"):
        with patch("builtins.exec", side_effect=lambda code, scope: scope.update({"evaluate": exploding_fn})):
            artifact_crash = gen._compile_aot_impl(g)
            res_recovered = artifact_crash(arr)
            np.testing.assert_allclose(res_recovered, arr + arr)
