"""Unit tests for the JAX code generator backend.

This module contains test cases that verify the correctness of the JAX code generation
process from a logical graph representation, including handling of basic operations,
shape broadcasting, unknown operations, and empty outputs.
"""

from ml_switcheroo_ir import LogicalGraph, LogicalNode

from ml_switcheroo_compiler.backends.jax import JAXCodeGenerator


def test_jax_generator_basic() -> None:
    """Verifies basic JAX code generation for a simple computational graph.

    This test constructs a logical graph containing inputs, a constant value,
    an addition operation, and an output node. It asserts that the generated
    JAX code correctly imports JAX, defines the model function, maps inputs,
    creates the constant array, performs the addition, and returns the result

    Returns:
    None
    """
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["in2"] = LogicalNode(id="in2", op_type="Input")
    graph.nodes["const1"] = LogicalNode(
        id="const1",
        op_type="Constant",
        attributes={"value": 42.0},
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
    """Verifies JAX code generation for shape expansion operations.

    This test constructs a logical graph containing a `BroadcastTo` operation
    and asserts that the generated JAX code correctly utilizes `jnp.broadcast_to`
    with the specified target shape metadata

    Returns:
    None
    """
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["expand"] = LogicalNode(
        id="expand",
        op_type="BroadcastTo",
        inputs=["in1"],
        shape_metadata=(1, 2, 3),
    )
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["expand"])

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = jnp.broadcast_to(input_0, (1, 2, 3))" in code


def test_jax_generator_unknown_op() -> None:
    """Verifies JAX code generation fallback behavior for unknown operations.

    This test constructs a logical graph containing an unrecognized operation
    type (`FooOp`) and asserts that the generator falls back to mapping it
    to a lowercase JAX function name (`jnp.fooop`)

    Returns:
    None
    """
    graph = LogicalGraph(name="test_jax", outputs=["out"])
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")
    graph.nodes["foo"] = LogicalNode(id="foo", op_type="FooOp", inputs=["in1"])
    graph.nodes["out"] = LogicalNode(id="out", op_type="Output", inputs=["foo"])

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "tensor_1 = jnp.fooop(input_0)" in code


def test_jax_generator_no_output() -> None:
    """Verifies JAX code generation when the logical graph has no output nodes.

    This test constructs a logical graph with only an input node and no
    explicit output nodes, asserting that the generated JAX code returns `None`

    Returns:
    None
    """
    graph = LogicalGraph(name="test_jax")
    graph.nodes["in1"] = LogicalNode(id="in1", op_type="Input")

    generator = JAXCodeGenerator(graph)
    code = generator.generate()

    assert "return None" in code
