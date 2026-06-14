"""Unit tests for the Keras code generator backend.

This module contains test cases that verify the correctness of the Keras code generation
process from a logical graph representation, including handling of inputs, constants,
standard operations, unknown operations, and empty outputs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.keras import KerasCodeGenerator


def test_keras_generator_basic() -> None:
    """Tests basic Keras code generation from a logical graph.

    This test constructs a simple logical graph with inputs, a constant, an addition
    operation, and an output, and verifies that the generated Keras code correctly
    defines the model structure, inputs, constants, operations, and outputs

    Args:
    None

    Returns:
    None
    """
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=(10, 20))
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "import keras" in code
    assert "def get_model():" in code
    assert "input_0 = keras.Input(shape=(10, 20), name='in1')" in code
    assert "input_1 = keras.Input(shape=(None,), name='in2')" in code
    assert "const_2 = 42.0" in code
    assert "tensor_3 = keras.ops.add(input_0, const_2)" in code
    assert "return keras.Model(inputs=[input_0, input_1], outputs=[tensor_3])" in code


def test_keras_generator_layer_map() -> None:
    """Tests Keras code generation for known layer operations.

    This test verifies that standard operations like Subtract, Multiply, and Relu
    are correctly mapped to their corresponding Keras operations in the generated
    code

    Args:
    None

    Returns:
    None
    """
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["sub"] = LogicalNode(
        id="sub",
        op_type="Subtract",
        inputs=["in1", "in2"],
    )
    graph.nodes["mul"] = LogicalNode(
        id="mul",
        op_type="Multiply",
        inputs=["sub", "in2"],
    )
    graph.nodes["relu"] = LogicalNode(id="relu", op_type="Relu", inputs=["mul"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["relu"])

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "keras.ops.subtract(input_0, input_1)" in code
    assert "keras.ops.multiply(tensor_2, input_1)" in code
    assert "keras.ops.relu(tensor_3)" in code


def test_keras_generator_unknown_op() -> None:
    """Tests Keras code generation fallback for unknown operations.

    This test ensures that when the logical graph contains an operation type that is
    not explicitly mapped, the generator falls back to a dynamic lowercase mapping
    under the `keras.ops` namespace

    Args:
    None

    Returns:
    None
    """
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooLayer", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = keras.ops.foolayer(input_0)" in code


def test_keras_generator_no_output() -> None:
    """Tests Keras code generation when no explicit outputs are defined.

    This test verifies that the generator can handle logical graphs with no defined
    outputs and correctly generates a Keras model with an empty outputs list

    Args:
    None

    Returns:
    None
    """
    graph = LogicalGraph(name="test_keras")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "return keras.Model(inputs=[input_0], outputs=[])" in code


def test_keras_generator_coverage() -> None:
    """Test keras generator coverage.

    Args:
    None

    Returns:
    None
    """
    gen = KerasCodeGenerator(LogicalGraph("foo"))

    class DummyNode:
        """Docstring."""

        op_type = "Matmul"

    res = gen.visit(DummyNode(), ["a", "b"], unrelated="hi")
    assert res == "keras.ops.matmul(a, b)"

    class DummyNode2:
        """Docstring."""

        op_type = "Zeros"

    res2 = gen.visit(DummyNode2(), ["a"], shape=[1], unrelated="hi")
    assert res2 == "keras.ops.zeros([1])"
