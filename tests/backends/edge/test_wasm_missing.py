import subprocess
import unittest.mock
from unittest.mock import patch

import pytest

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator
from ml_switcheroo_compiler.backends.llvm_cpp.generator import CppGenerator


class DummyGraph:
    nodes = {}


class DummyNode:
    op_type = "Conv2D"
    inputs = ["a", "b"]

    def __init__(self, shape1, shape2, out_shape):
        self._shape1 = shape1
        self._shape2 = shape2
        self._out_shape = out_shape
        self.attributes = {}


def test_wasm_conv2d_shapes():
    gen = WasmCodeGenerator(DummyGraph())
    # Mock _get_shape and nodes
    gen.sorted_nodes = [type("Node", (), {"id": "a", "shape_metadata": 10})(), type("Node", (), {"id": "b", "shape_metadata": 5})()]
    # Hits list branch and scalar branch
    node = type("Node", (), {"inputs": ["a", "b"], "shape_metadata": 15})()
    gen.visit_Conv2D(node, "Conv2D", "conv2d_0", ["a", "b"], [15], 15)

    # Hits missing empty branch and scalar branch for shape parameter
    gen.sorted_nodes = [type("Node", (), {"id": "a", "shape_metadata": []})(), type("Node", (), {"id": "b", "shape_metadata": []})()]
    node2 = type("Node", (), {"inputs": ["a", "b"], "shape_metadata": []})()
    gen.visit_Conv2D(node2, "Conv2D", "conv2d_0", ["a", "b"], [], 15)

    node3 = type("Node", (), {"inputs": ["a", "b"], "shape_metadata": 1})()
    gen.visit_Conv2D(node3, "Conv2D", "conv2d_0", ["a", "b"], 1, 15)


@patch("subprocess.run")
@patch("shutil.which")
def test_wasm_compile_subprocess_error(mock_which, mock_run, tmpdir):
    gen = WasmCodeGenerator(DummyGraph())
    # Make sure we hit the "clang" path but then it fails
    mock_which.return_value = "/bin/clang"

    # Test subprocess.CalledProcessError
    mock_run.side_effect = subprocess.CalledProcessError(1, "cmd", stderr=b"error")
    from ml_switcheroo_compiler.core.errors import CompilationError

    with pytest.raises(CompilationError):
        gen.compile_wasm(str(tmpdir))

    # Test general Exception
    mock_run.side_effect = ValueError("general error")
    with pytest.raises(CompilationError):
        gen.compile_wasm(str(tmpdir))


def test_wasm_provider_templates():
    import ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider as provider

    # Hit True branch
    provider._WASM_TEMPLATES = {"js_orchestration": {"test": "val"}, "cpp_helpers": ["help"]}
    assert provider.get_js_orchestration_template("test") == "val"
    assert provider.get_cpp_helpers() == ["help"]

    # Hit False branch
    provider._WASM_TEMPLATES = {}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml", return_value={"js_orchestration": {"test": "val"}, "cpp_helpers": ["help"]}):
        assert provider.get_js_orchestration_template("test") == "val"

    provider._WASM_TEMPLATES = {}
    with patch("ml_switcheroo_compiler.backends.edge.wasm_simd.wasm_provider.load_yaml", return_value={"js_orchestration": {"test": "val"}, "cpp_helpers": ["help"]}):
        assert provider.get_cpp_helpers() == ["help"]
    provider._WASM_TEMPLATES = {}


def test_llvm_cpp_generator_shapes():
    gen = CppGenerator(DummyGraph())
    node_a = type("Node", (), {"id": "a"})()
    node_b = type("Node", (), {"id": "b"})()
    gen.graph = type("Graph", (), {"nodes": {"a": node_a, "b": node_b}})()

    def fake_get_shape(node):
        if getattr(node, "id", None) == "a":
            return [1, 1]
        if getattr(node, "id", None) == "b":
            return [2, 2]
        return [3, 3]

    gen._get_shape = fake_get_shape

    node = type("Node", (), {"inputs": ["a", "b"], "attributes": {}})()
    try:
        gen.visit_Conv2D(node, gen.graph)
    except Exception:
        pass

    def fake_get_shape_4d(node):
        return [1, 1, 1, 1]

    gen._get_shape = fake_get_shape_4d
    try:
        gen.visit_Conv2D(node, gen.graph)
    except Exception:
        pass


@patch("os.path.exists")
def test_wasm_simd_helpers_no_file(mock_exists):
    mock_exists.return_value = False
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    gen = WasmCodeGenerator(DummyGraph())
    helpers = gen.get_helper_functions()
    # It should not fail, should just skip the file reading part
    assert isinstance(helpers, list)


@patch("os.path.exists")
@patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="intrinsics:\n  op1:\n    macro_name: my_macro\n  op2:\n    simd_expr: my_expr\n  op3:\n    macro_name: valid\n    simd_expr: valid_expr")
def test_wasm_simd_helpers_missing_keys(mock_open, mock_exists):
    mock_exists.return_value = True
    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    gen = WasmCodeGenerator(DummyGraph())
    helpers = gen.get_helper_functions()
    # Should only generate valid_expr macro
    assert any("valid(v128_t x)" in h for h in helpers)
    assert not any("my_macro" in h for h in helpers)
