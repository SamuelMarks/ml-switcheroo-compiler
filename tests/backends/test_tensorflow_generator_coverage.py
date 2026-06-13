"""Module docstring."""

from ml_switcheroo_compiler.backends.tensorflow import TensorFlowCodeGenerator
from ml_switcheroo_compiler.ir.core import IRNode, IRGraph


def test_tensorflow_generator_coverage_brute() -> None:
    """Function docstring."""
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)

    # Test ops_map missing kwarg placeholder branch line 83->82 loop continue
    node5 = IRNode(id="n5", op_type="Zeros", inputs=[], attributes={"fake": 1}, shape_metadata=None)
    res5 = gen.visit(node5, [], shape="(2, 2)", fake=1)
    assert res5 == "tf.zeros((2, 2))"
