"""Module docstring."""

from ml_switcheroo_compiler.backends.mlx import MLXCodeGenerator
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_mlx_generator_coverage_brute() -> None:
    """Function docstring."""
    g = IRGraph()
    gen = MLXCodeGenerator(g)

    # Test ops_map missing kwarg placeholder branch line 44->43 loop continue
    node1 = IRNode(id="n1", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
    res1 = gen.visit(node1, [], shape="(2, 2)", fake=1)
    assert res1 == "mx.zeros((2, 2))"
