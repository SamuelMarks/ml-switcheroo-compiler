from ml_switcheroo_compiler.backends.cuda.cuda import CudaCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_cuda_generator():
    """Test CUDA code generation."""
    graph = IRGraph()
    graph.nodes = {"matmul": IRNode(id="matmul", op_type="MatMul", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" in out


def test_cuda_generator_coverage():
    """Test CUDA generator coverage."""
    graph = IRGraph()
    graph.nodes = {"input": IRNode(id="input", op_type="Input", inputs=[])}
    gen = CudaCodeGenerator(graph)
    out = gen.generate()
    assert "matmul" not in out


def test_cuda_missing_yaml(monkeypatch):
    import os

    monkeypatch.setattr(os.path, "exists", lambda p: False)
    graph = IRGraph()
    gen = CudaCodeGenerator(graph)
    assert gen.config.templates == {}
