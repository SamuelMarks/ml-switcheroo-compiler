"""Test WASM backend edge cases coverage."""

from unittest.mock import MagicMock

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator


def test_wasm_coverage():
    """Test WASM code generation edge cases."""
    g = LogicalGraph(outputs=["n_simd_add", "n_simd_sub", "n_simd_mul", "n_simd_div", "n_simd_exp", "n_simd_neg", "n_simd_generic"])

    # Simd
    n_simd_input1 = LogicalNode(id="n_simd_input1", op_type="Input")
    n_simd_input2 = LogicalNode(id="n_simd_input2", op_type="Input")

    n_simd_add = LogicalNode(id="n_simd_add", op_type="Add", inputs=["n_simd_input1", "n_simd_input2"])
    n_simd_sub = LogicalNode(id="n_simd_sub", op_type="Subtract", inputs=["n_simd_input1", "n_simd_input2"])
    n_simd_mul = LogicalNode(id="n_simd_mul", op_type="Multiply", inputs=["n_simd_input1", "n_simd_input2"])
    n_simd_div = LogicalNode(id="n_simd_div", op_type="TrueDivide", inputs=["n_simd_input1", "n_simd_input2"])
    n_simd_exp = LogicalNode(id="n_simd_exp", op_type="Exp", inputs=["n_simd_input1"])
    n_simd_neg = LogicalNode(id="n_simd_neg", op_type="Negative", inputs=["n_simd_input1"])
    n_simd_generic = LogicalNode(id="n_simd_generic", op_type="Tan", inputs=["n_simd_input1"])
    n_simd_const = LogicalNode(id="n_simd_const", op_type="Constant", attributes={"value": 1.0})

    g.nodes = {
        "n_simd_input1": n_simd_input1,
        "n_simd_input2": n_simd_input2,
        "n_simd_add": n_simd_add,
        "n_simd_sub": n_simd_sub,
        "n_simd_mul": n_simd_mul,
        "n_simd_div": n_simd_div,
        "n_simd_exp": n_simd_exp,
        "n_simd_neg": n_simd_neg,
        "n_simd_generic": n_simd_generic,
        "n_simd_const": n_simd_const,
    }

    gen = WasmCodeGenerator(g)

    # Check evaluate_node None
    assert gen.generic_visit(None, []) == ""

    # Check SIMD mode paths
    gen.is_simd = True
    assert gen.generic_visit(n_simd_const, []) == "v_n_simd_const_simd"
    assert gen.generic_visit(n_simd_add, []) == "v_n_simd_add_simd"
    assert gen.generic_visit(n_simd_sub, []) == "v_n_simd_sub_simd"
    assert gen.generic_visit(n_simd_mul, []) == "v_n_simd_mul_simd"
    assert gen.generic_visit(n_simd_div, []) == "v_n_simd_div_simd"
    assert gen.generic_visit(n_simd_exp, []) == "v_n_simd_exp_simd"
    assert gen.generic_visit(n_simd_neg, []) == "v_n_simd_neg_simd"
    assert gen.generic_visit(n_simd_generic, []) == "v_n_simd_generic_simd"


def test_wasm_compile_coverage(monkeypatch):
    """Test WASM compile function."""
    import tempfile

    g = LogicalGraph(outputs=["n2"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    g.nodes["n2"] = LogicalNode(id="n2", op_type="Negative", inputs=["n1"])
    gen = WasmCodeGenerator(g)

    # We need to monkeypatch tempfile and subprocess so we don't actually leave temp files
    # and hit the compile branches

    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    class MockTempFile:
        def __init__(self, *args, **kwargs):
            self.name = "/tmp/dummy.cpp"

        def write(self, data):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", MockTempFile)

    # Emcc fallback success
    monkeypatch.setattr("shutil.which", lambda cmd: "/bin/emcc" if cmd == "emcc" else None)
    monkeypatch.setattr("os.path.exists", lambda p: True)
    monkeypatch.setattr("os.remove", lambda p: None)
    res = gen.compile_wasm("/dummy_dir")
    assert res is not None
    assert "/dummy_dir/kernel.js" in res[0]

    # Clang fallback success
    monkeypatch.setattr("shutil.which", lambda cmd: "/bin/clang" if cmd == "clang" else None)
    res = gen.compile_wasm("/dummy_dir")
    assert res is not None
    assert res[0] == ""
    assert "/dummy_dir/kernel.wasm" in res[1]

    # Exception fallback inside compile
    mock_run.side_effect = Exception("test error")
    res = gen.compile_wasm("/dummy_dir")
    assert res is None


def test_wasm_scalar_fallback():
    """Test WASM scalar code generation edge cases."""
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    g = LogicalGraph(outputs=["n2"])

    n_scalar_input = LogicalNode(id="n_scalar_input", op_type="Input")
    n_scalar_generic = LogicalNode(id="n_scalar_generic", op_type="UnknownOp", inputs=["n_scalar_input"])

    g.nodes = {
        "n_scalar_input": n_scalar_input,
        "n_scalar_generic": n_scalar_generic,
    }

    gen = WasmCodeGenerator(g)

    gen.is_simd = False
    assert gen.generic_visit(n_scalar_generic, ["n_scalar_input"]) == "v_n_scalar_generic_scalar"
    assert "unknownop(" in gen.body_lines_scalar[-1]


def test_wasm_compile_clang_fallback(monkeypatch):
    """Test WASM compile function clang branch and file removal."""
    import os
    import tempfile

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    gen = WasmCodeGenerator(g)

    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    class MockTempFile:
        def __init__(self, *args, **kwargs):
            self.name = "/tmp/dummy2.cpp"

        def write(self, data):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", MockTempFile)

    # We must properly trick the file exist branch into True so remove is called
    def mock_exists(p):
        return p == "/tmp/dummy2.cpp"

    monkeypatch.setattr(os.path, "exists", mock_exists)

    mock_remove = MagicMock()
    monkeypatch.setattr(os, "remove", mock_remove)

    # Clang fallback success
    monkeypatch.setattr("shutil.which", lambda cmd: "/bin/clang" if cmd == "clang" else None)
    res = gen.compile_wasm("/dummy_dir")
    assert res is not None
    assert res[0] == ""
    assert "/dummy_dir/kernel.wasm" in res[1]

    # Verify os.remove was called
    mock_remove.assert_called_once_with("/tmp/dummy2.cpp")


def test_wasm_compile_clang_fallback_fail(monkeypatch):
    """Test WASM compile function clang branch failure."""
    import tempfile

    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    from ml_switcheroo_compiler.backends.edge.wasm import WasmCodeGenerator

    g = LogicalGraph(outputs=["n1"])
    g.nodes["n1"] = LogicalNode(id="n1", op_type="Input")
    gen = WasmCodeGenerator(g)

    mock_run = MagicMock()
    monkeypatch.setattr("subprocess.run", mock_run)

    class MockTempFile:
        def __init__(self, *args, **kwargs):
            self.name = "/tmp/dummy2.cpp"

        def write(self, data):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", MockTempFile)

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr("os.path.exists", lambda p: False)

    res = gen.compile_wasm("/dummy_dir")
    assert res is None
