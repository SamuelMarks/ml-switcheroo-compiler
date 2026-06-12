"""Tests for JAX target code generation."""

from ml_switcheroo.backends.jax import JAXCodeGenerator
from ml_switcheroo_ir import LogicalGraph, LogicalNode


def test_jax_generator_basic() -> None:
    """Test basic JAX code generation."""
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1", op_type="Constant", attributes={"value": 42.0}
    )
    graph.nodes["add"] = LogicalNode(id="add", op_type="Add", inputs=["in1", "const1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["add"])

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "import jax.numpy as jnp" in code
    assert "def apply_model(params, *args, **kwargs):" in code
    assert "input_0 = args[0]" in code
    assert "input_1 = args[1]" in code
    assert "const_2 = jnp.array(42.0)" in code
    assert "tensor_3 = jnp.add(input_0, const_2)" in code
    assert "return tensor_3" in code


def test_jax_generator_expand_shape() -> None:
    """Test JAX code generation with Expand op."""
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand", op_type="BroadcastTo", inputs=["in1"], shape_metadata=(1, 2, 3)
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = jnp.broadcast_to(input_0, (1, 2, 3))" in code


def test_jax_generator_unknown_op() -> None:
    """Test JAX code generation falls back to lower-case op mapping."""
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = unknown_op_fooop(input_0)" in code


def test_jax_generator_no_output() -> None:
    """Test JAX code generation without explicit output node."""
    graph = LogicalGraph(name="test_jax")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
