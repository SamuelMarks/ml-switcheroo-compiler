from ml_switcheroo_compiler.backends.metal.metal import MetalCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_metal_generator():
    """Test Metal code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = MetalCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_metal_generator_coverage():
    """Test Metal generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = MetalCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_metal_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = MetalCodeGenerator(graph)
    assert gen.config.templates == {}
