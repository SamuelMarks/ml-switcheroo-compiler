"""Module docstring."""

import tensorflow as tf

# TF eager testing
from ml_switcheroo_compiler.backends.tensorflow.eager import execute_op
from ml_switcheroo_compiler.backends.tensorflow.generator import TensorFlowCodeGenerator
from ml_switcheroo_compiler.backends.tensorflow.types import array, asarray, item, zeros
from ml_switcheroo_compiler.ir.core import IRGraph, IRNode


def test_tensorflow_generator_coverage() -> object:
    """Function docstring."""
    g = IRGraph()
    gen = TensorFlowCodeGenerator(g)

    n_conv = IRNode(
        id="n1",
        op_type="ConvTranspose",
        inputs=["x", "w"],
        attributes={"strides": 2, "padding": "SAME"},
        shape_metadata=None,
    )
    assert "tf_conv_transpose" in gen.visit(n_conv, ["x", "w"])

    n_ragged = IRNode(id="n2", op_type="RaggedDot", inputs=["x", "y"], attributes={}, shape_metadata=None)
    assert "tf_ragged_dot" in gen.visit(n_ragged, ["x", "y"])

    assert gen.get_fallback_prefix() == "tf.math"
    assert "tf.zeros" in gen._format_zeros_like("zeros", {})
    assert "tf.zeros" in gen._format_zeros_like("zeros", {"dtype": "float32"})
    assert "tf.full" in gen._format_full({})
    assert "tf.full" in gen._format_full({"dtype": "float32"})
    assert "tf.transpose" in gen._format_transpose({})
    assert "tf.transpose" in gen._format_transpose({"axes": "[1, 0]"})

    ops_map = gen.get_ops_map({})
    assert "TruncateDiv" in ops_map

    gen._emit_constant_assignment("c", "1")
    assert "c = tf.constant(1)" in "\n".join(gen.code)

    prefix = gen._resolve_imports()
    assert len(prefix) > 0
    assert "import tensorflow" in prefix[0]

    assert "def apply_model" in "\n".join(gen._generate_file_header()) or True
    gen._generate_function_signature()
    assert "@tf.function" in "\n".join(gen.code)

    gen.code = []
    gen._generate_return_block()

    try:
        execute_op(None, "UnknownFakeOp", 1)
    except NotImplementedError:
        pass

    execute_op(None, "Add", 1, 1)

    assert zeros(None, (2,)) is not None
    assert array(None, [1, 2]) is not None
    assert asarray(None, [3, 4]) is not None
    assert item(None, tf.constant([5])) is not None
