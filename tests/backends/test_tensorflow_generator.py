"""Unit tests for the TensorFlow code generator.

This module contains tests that verify the correctness of the TensorFlowCodeGenerator
class, ensuring it correctly translates logical graphs into executable TensorFlow code
for various operations, edge cases, and fallback scenarios.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo.backends.tensorflow import TensorFlowCodeGenerator


def test_tensorflow_generator_basic() -> None:
    """Tests basic TensorFlow code generation for a simple computational graph.

    Verifies that the TensorFlowCodeGenerator correctly handles inputs, constants,
    addition operations, and outputs, producing a valid TensorFlow function with
    appropriate imports and decorators

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
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
    assert "tensor_3 = tf.math.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_tensorflow_generator_expand_shape() -> None:
    """Tests TensorFlow code generation for the BroadcastTo (Expand) operation.

    Verifies that the generator correctly translates a BroadcastTo logical node
    into a tf.broadcast_to call with the specified target shape metadata

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand",
        op_type="BroadcastTo",
        inputs=["in1"],
        shape_metadata=(1, 2, 3),
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = tf.broadcast_to(input_0, (1, 2, 3))" in code


def test_tensorflow_generator_unknown_op() -> None:
    """Tests TensorFlow code generation fallback behavior for unknown operations.

    Verifies that when the generator encounters an unrecognized operation type,
    it falls back to mapping it to a raw or standard tf.math operation

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = tf.math.fooop(input_0)" in code


def test_tensorflow_generator_no_output() -> None:
    """Tests TensorFlow code generation for a graph with no explicit output nodes.

    Verifies that the generator handles graphs without defined outputs gracefully
    by generating a function that returns None

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
