from ml_switcheroo_compiler.backends.edge.webgl import WebGLCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_webgl_generator():
    """Test WebGL code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_matmul" in out


def test_webgl_generator_coverage():
    """Test WebGL generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_input" not in out


def test_webgl_generator_shapes_and_inputs():
    graph = IRGraph()
    n0 = IRNode(id="n0", op_type="Input", inputs=[], shape_metadata=(16, 32))
    n1 = IRNode(id="n1", op_type="Input", inputs=[], shape_metadata=(32, 64))
    n2 = IRNode(id="n2", op_type="MatMul", inputs=["n0", "n1"], shape_metadata=(16, 64))
    n3 = IRNode(id="n3", op_type="Add", inputs=["n2", "n2"], shape_metadata=(16, 64))
    n4 = IRNode(id="n4", op_type="Exp", inputs=["n3"], shape_metadata=(64,))
    graph.nodes = {"n0": n0, "n1": n1, "n2": n2, "n3": n3, "n4": n4}
    gen = WebGLCodeGenerator(graph)
    out = gen.generate()
    assert "shader_n2" in out
    assert "shader_n3" in out
    assert "shader_n4" in out
    assert "getUniformLocation" in out


def test_webgl_generator_missing_op():
    import pytest

    graph = IRGraph()
    n = IRNode(id="n", op_type="MissingOp", inputs=[])
    graph.nodes = {"n": n}
    gen = WebGLCodeGenerator(graph)
    with pytest.raises(ValueError, match="Missing WebGL shader template for operation: MissingOp"):
        gen.generate()


def test_webgl_generator_custom_setup():
    graph = IRGraph()
    n = IRNode(id="n", op_type="Add", inputs=[], shape_metadata=(16, 32))
    graph.nodes = {"n": n}
    gen = WebGLCodeGenerator(graph)
    from ml_switcheroo_compiler.backends.edge.config_models import WebglTemplateConfig

    gen.config.templates["add"] = WebglTemplateConfig(body="test", custom_setup="let test = {k_dim};")
    out = gen.generate()
    assert "let test = 32;" in out

    gen.config.templates["add"] = "test_string_body"
    out2 = gen.generate()
    assert "test_string_body" in out2


def test_webgl_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = WebGLCodeGenerator(graph)
    assert gen.config.templates == {}
