"""Unit tests for the TensorFlow code generator.

This module contains tests that verify the correctness of the TensorFlowCodeGenerator
class, ensuring it correctly translates logical graphs into executable TensorFlow code
for various operations, edge cases, and fallback scenarios.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.tensorflow import TensorFlowCodeGenerator


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


def test_tensorflow_generator_ops_map_kwargs() -> None:
    """Tests kwargs replacement in ops_map operations.

    Verifies that the generator correctly replaces or strips 'axis' and 'keepdims'
    kwargs for operations defined in ops_map like 'Sum'.

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf", outputs=["out1", "out2"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    # Missing keepdims and axis
    graph.nodes["sum1"] = LogicalNode(id="sum1", op_type="Sum", inputs=["in1"])
    # With keepdims and axis
    graph.nodes["sum2"] = LogicalNode(
        id="sum2", op_type="Sum", inputs=["in1"], attributes={"axis": 0, "keepdims": True}
    )
    graph.nodes["out1"] = LogicalNode(id="out1", op_type="Output", inputs=["sum1"])
    graph.nodes["out2"] = LogicalNode(id="out2", op_type="Output", inputs=["sum2"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tf.reduce_sum(input_0)" in code
    assert "tf.reduce_sum(input_0, axis=0, keepdims=True)" in code


def test_tensorflow_generator_generic_kwargs() -> None:
    """Tests kwargs fallback for generic operations.

    Verifies that 'axis' and 'keepdims' are appended to the argument list
    for generic operations not defined in ops_map.

    Returns:
    None
    """
    graph = LogicalGraph(name="test_tf", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(
        id="foo", op_type="FooOp", inputs=["in1"], attributes={"axis": 1, "keepdims": True}
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = TensorFlowCodeGenerator(graph)
    code = generator.generate()

    assert "tf.math.fooop(input_0, axis=1, keepdims=True)" in code
