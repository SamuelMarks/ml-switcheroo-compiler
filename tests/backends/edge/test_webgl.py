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


def test_webgl_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = WebGLCodeGenerator(graph)
    assert gen.config.templates == {}
