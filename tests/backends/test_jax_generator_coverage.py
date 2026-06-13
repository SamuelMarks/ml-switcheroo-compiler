"""Provides required module functionality."""

from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator
from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_jax_generator_coverage_brute() -> None:
    """Execute the requested function."""
    g = IRGraph()
    gen = JAXCodeGenerator(g)

    node1 = IRNode(
        id="n1",
        op_type="Zeros",
        inputs=[],
        attributes={"fake": 1, "shape": "(2, 2)", "fake2": 2},
        shape_metadata=None,
    )
    gen.visit(node1, [], shape="(2, 2)", fake=1, fake2=2)

    gen_mlx = MLXCodeGenerator(g)
    gen_mlx.visit(node1, [], shape="(2, 2)", fake=1, fake2=2)
