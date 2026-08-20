from ml_switcheroo_compiler.backends.rocm.rocm import RocmCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_rocm_generator():
    """Test ROCm code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_rocm_generator_coverage():
    """Test ROCm generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = RocmCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_rocm_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = RocmCodeGenerator(graph)
    assert gen.config.templates == {}
