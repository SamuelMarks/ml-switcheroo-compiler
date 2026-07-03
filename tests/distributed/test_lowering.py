"""Module docstring."""

from ml_switcheroo_compiler.backends.jax.generator import JAXCodeGenerator
from ml_switcheroo_compiler.backends.pytorch.generator import PyTorchCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_jax_spmd_lowering() -> object:
    """Function docstring."""
    graph = IRGraph()
    gen = JAXCodeGenerator(graph)
    node_gather = IRNode(id="n1", op_type="all_gather", inputs=["x"])
    code_gather = gen.visit(node_gather, ["x"])
    assert "jax.lax.all_gather" in code_gather

    node_scatter = IRNode(id="n2", op_type="reduce_scatter", inputs=["x"])
    code_scatter = gen.visit(node_scatter, ["x"])
    assert "jax.lax.reduce_scatter" in code_scatter

    node_allreduce = IRNode(id="n3", op_type="all_reduce", inputs=["x"])
    code_allreduce = gen.visit(node_allreduce, ["x"])
    assert "jax.lax.pmean" in code_allreduce or "jax.lax.psum" in code_allreduce


def test_torch_spmd_lowering() -> object:
    """Function docstring."""
    graph = IRGraph()
    gen = PyTorchCodeGenerator(graph)
    node_gather = IRNode(id="n1", op_type="all_gather", inputs=["x"])
    code_gather = gen.visit(node_gather, ["x"])
    assert "torch.distributed.all_gather_into_tensor" in code_gather or "torch.distributed.all_gather" in code_gather or "dist.all_gather" in code_gather

    node_scatter = IRNode(id="n2", op_type="reduce_scatter", inputs=["x"])
    code_scatter = gen.visit(node_scatter, ["x"])
    assert "torch.distributed.reduce_scatter_tensor" in code_scatter or "torch.distributed.reduce_scatter" in code_scatter or "dist.reduce_scatter" in code_scatter

    node_allreduce = IRNode(id="n3", op_type="all_reduce", inputs=["x"])
    code_allreduce = gen.visit(node_allreduce, ["x"])
    assert "torch.distributed.all_reduce" in code_allreduce or "dist.all_reduce" in code_allreduce
