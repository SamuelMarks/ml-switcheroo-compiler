"""Tests for TensorFlow target code generation."""

from ml_switcheroo.backends.tensorflow import TensorFlowCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_tensorflow_generator_basic() -> None:
    """Test basic TensorFlow code generation."""
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "import tensorflow as tf" in code
    assert "@tf.function" in code
    assert "def apply_model(*args, **kwargs):" in code
    assert "input_0 = args[0]" in code
    assert "input_1 = args[1]" in code
    assert "const_2 = tf.constant(42.0)" in code
    assert "tensor_3 = tf.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_tensorflow_generator_expand_shape() -> None:
    """Test TensorFlow code generation with Expand op."""
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand", op_type="Expand", inputs=["in1"], shape_metadata=(1, 2, 3)
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = tf.broadcast_to(input_0, shape=(1, 2, 3))" in code


def test_tensorflow_generator_unknown_op() -> None:
    """Test TensorFlow code generation falls back to raw_ops mapping."""
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = tf.raw_ops.FooOp(input_0)" in code


def test_tensorflow_generator_no_output() -> None:
    """Test TensorFlow code generation without explicit output node."""
    graph = LogicalGraph(name="test_tf")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
