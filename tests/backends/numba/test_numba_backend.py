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
