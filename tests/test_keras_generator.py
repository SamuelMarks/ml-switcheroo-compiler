"""Tests for Keras target code generation."""

from ml_switcheroo.backends.keras import KerasCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_keras_generator_basic() -> None:
    """Test basic Keras code generation."""
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input", shape_metadata=(10, 20))
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
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
    assert "tensor_3 = keras.layers.Add()([input_0, const_2])" in code
    assert "return keras.Model(inputs=[input_0, input_1], outputs=[tensor_3])" in code


def test_keras_generator_layer_map() -> None:
    """Test Keras code generation for known layers."""
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["sub"] = LogicalNode(id="sub", op_type="Sub", inputs=["in1", "in2"])
    graph.nodes["mul"] = LogicalNode(id="mul", op_type="Mul", inputs=["sub", "in2"])
    graph.nodes["relu"] = LogicalNode(id="relu", op_type="Relu", inputs=["mul"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["relu"])

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "keras.layers.Subtract()([input_0, input_1])" in code
    assert "keras.layers.Multiply()([tensor_2, input_1])" in code
    assert "keras.layers.ReLU()(tensor_3)" in code


def test_keras_generator_unknown_op() -> None:
    """Test Keras code generation falls back to dynamic layer mapping."""
    graph = LogicalGraph(name="test_keras", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooLayer", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = keras.layers.FooLayer()(input_0)" in code


def test_keras_generator_no_output() -> None:
    """Test Keras code generation without explicit output node."""
    graph = LogicalGraph(name="test_keras")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = KerasCodeGenerator(graph)
    code = generator.generate()

    assert "return keras.Model(inputs=[input_0], outputs=[])" in code
